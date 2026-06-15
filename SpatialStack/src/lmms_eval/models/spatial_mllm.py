import importlib
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPATIAL_MLLM_REPO_NAME = "Spatial-MLLM"


def _is_video_path(path: str) -> bool:
    return path.lower().endswith(VIDEO_EXTENSIONS)


def _is_image_path(path: str) -> bool:
    return path.lower().endswith(IMAGE_EXTENSIONS)


def _load_rgb_image(image: Union[str, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    raise NotImplementedError(f"Unsupported image type: {type(image)}")


def _normalize_sample_visual(raw_visual):
    if raw_visual is None:
        return None
    if isinstance(raw_visual, Image.Image):
        return raw_visual.convert("RGB")
    if isinstance(raw_visual, str):
        return raw_visual
    if isinstance(raw_visual, (list, tuple)):
        visuals = []
        for visual in raw_visual:
            if isinstance(visual, str) and _is_video_path(visual):
                if len(raw_visual) != 1:
                    raise NotImplementedError("Mixed video lists are not supported.")
                return visual
            visuals.append(_load_rgb_image(visual))
        return visuals
    raise NotImplementedError(f"Unsupported visual type: {type(raw_visual)}")


def _prepare_spatial_mllm_inputs(batch, video_inputs, image_inputs):
    video_tchw = []
    image_tchw = []

    if video_inputs:
        for video_input in video_inputs:
            if isinstance(video_input, torch.Tensor):
                video_input = video_input.float() / 255.0
            elif isinstance(video_input, list) and all(isinstance(img, Image.Image) for img in video_input):
                video_input = torch.stack(
                    [torch.tensor(np.array(img)).permute(2, 0, 1) for img in video_input]
                ).float() / 255.0
            else:
                raise ValueError(f"Unsupported video input format: {type(video_input)}")
            video_tchw.append(video_input)

    if image_inputs:
        for image_input in image_inputs:
            if isinstance(image_input, Image.Image):
                image_input = torch.tensor(np.array(image_input)).permute(2, 0, 1).float() / 255.0
            else:
                raise ValueError(f"Unsupported image input format: {type(image_input)}")
            image_tchw.append(image_input)

    batch.update(
        {
            "video_tchw": video_tchw if video_tchw else None,
            "image_tchw": image_tchw if image_tchw else None,
        }
    )
    return batch


def _get_spatialstack_omega_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _get_default_spatial_mllm_repo_path() -> Path:
    return _get_spatialstack_omega_root().parent / SPATIAL_MLLM_REPO_NAME


def _resolve_spatial_mllm_repo_path(spatial_mllm_repo_path: Optional[str]) -> Path:
    if spatial_mllm_repo_path in (None, "", "auto"):
        return _get_default_spatial_mllm_repo_path()

    candidate = Path(os.path.expanduser(spatial_mllm_repo_path))
    if candidate.is_absolute():
        return candidate

    # Interpret relative paths against the SpatialStack-omega parent directory,
    # so `Spatial-MLLM` resolves to a sibling checkout on local and remote hosts.
    return _get_spatialstack_omega_root().parent / candidate


def _import_spatial_mllm_modules(spatial_mllm_repo_path: Optional[str]):
    repo_path = _resolve_spatial_mllm_repo_path(spatial_mllm_repo_path)
    if not os.path.isdir(repo_path):
        raise FileNotFoundError(
            f"Spatial-MLLM repo path does not exist: {repo_path}. "
            f"Expected the default sibling checkout at {_get_default_spatial_mllm_repo_path()}. "
            "Override with `spatial_mllm_repo_path=...` in MODEL_ARGS_BASE/MODEL_ARGS_EXTRA if needed."
        )

    repo_path_str = str(repo_path)
    if repo_path_str not in sys.path:
        sys.path.insert(0, repo_path_str)
    external_root = repo_path / "src" / "qwenvl" / "external"
    external_root_str = str(external_root)
    if external_root.is_dir() and external_root_str not in sys.path:
        sys.path.insert(0, external_root_str)

    spatial_mllm_module = importlib.import_module("src.qwenvl.model.spatial_mllm")
    return spatial_mllm_module.SpatialMLLMConfig, spatial_mllm_module.SpatialMLLMForConditionalGeneration


@register_model("spatial_mllm")
class SpatialMLLM(lmms):
    def __init__(
        self,
        pretrained: str = "Diankun/Spatial-MLLM-v1.1-Instruct-135K",
        spatial_mllm_repo_path: Optional[str] = None,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        use_flash_attention_2: Optional[bool] = True,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 16,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        SpatialMLLMConfig, SpatialMLLMForConditionalGeneration = _import_spatial_mllm_modules(spatial_mllm_repo_path)
        try:
            from transformers import Qwen2_5_VLProcessor
        except ImportError as exc:
            raise RuntimeError("Spatial-MLLM eval requires transformers with Qwen2.5-VL support.") from exc

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = str(self._device)
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        config = SpatialMLLMConfig.from_pretrained(pretrained)
        load_kwargs = {
            "config": config,
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
        }
        if use_flash_attention_2:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        self._model = SpatialMLLMForConditionalGeneration.from_pretrained(pretrained, **load_kwargs).eval()

        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            pretrained,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            padding_side="left",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")
        if max_length is not None:
            setattr(self.processor.tokenizer, "model_max_length", max_length)
            setattr(self._tokenizer, "model_max_length", max_length)

        self.max_num_frames = max_num_frames
        self.use_cache = use_cache
        self.batch_size_per_gpu = int(batch_size)
        self._config = self.model.config
        self._max_length = getattr(self._tokenizer, "model_max_length", None)

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Spatial-MLLM.")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            sample_visuals = [_normalize_sample_visual(doc_to_visual[0](self.task_dict[task][split][ids])) for ids in doc_id]

            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be Union[str, list], got {type(until)}")

            messages = []
            for visual, context in zip(sample_visuals, contexts):
                user_content = []
                if isinstance(visual, str) and _is_video_path(visual):
                    user_content.append({"type": "video", "video": visual, "nframes": self.max_num_frames})
                elif isinstance(visual, str) and _is_image_path(visual):
                    user_content.append({"type": "image", "image": visual})
                elif isinstance(visual, Image.Image):
                    user_content.append({"type": "image", "image": visual})
                elif isinstance(visual, list) and all(isinstance(v, Image.Image) for v in visual):
                    for image in visual:
                        user_content.append({"type": "image", "image": image})
                elif visual is not None:
                    raise NotImplementedError(f"Unsupported normalized visual type: {type(visual)}")

                user_content.append({"type": "text", "text": context})
                messages.append([{"role": "user", "content": user_content}])

            text = [
                self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                for message in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=text,
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                return_tensors="pt",
                padding=True,
                padding_side="left",
            )
            inputs = _prepare_spatial_mllm_inputs(inputs, video_inputs, image_inputs)

            device = "cuda" if self.device_map == "auto" else self.device
            inputs = inputs.to(device)
            if inputs.get("image_tchw") is not None:
                inputs["image_tchw"] = [image_tchw.to(device) for image_tchw in inputs["image_tchw"]]
            if inputs.get("video_tchw") is not None:
                inputs["video_tchw"] = [video_tchw.to(device) for video_tchw in inputs["video_tchw"]]

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0.1
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = 0.001
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            output_ids = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for answer, context in zip(answers, contexts):
                res.append(answer)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
