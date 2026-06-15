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


def _convert_image_list_to_tchw(image_inputs):
    tensors = []
    for image_input in image_inputs:
        if not isinstance(image_input, Image.Image):
            raise ValueError(f"Unsupported image input format: {type(image_input)}")
        tensors.append(torch.tensor(np.array(image_input)).permute(2, 0, 1).float() / 255.0)
    return torch.stack(tensors, dim=0)


def _convert_video_input_to_tchw(video_input):
    if isinstance(video_input, torch.Tensor):
        if video_input.ndim == 4:
            if video_input.shape[-1] in (1, 3):
                video_input = video_input.permute(0, 3, 1, 2)
            return video_input.float() / 255.0
        raise ValueError(f"Unsupported video tensor shape: {tuple(video_input.shape)}")
    if isinstance(video_input, list) and all(isinstance(img, Image.Image) for img in video_input):
        return torch.stack([torch.tensor(np.array(img)).permute(2, 0, 1) for img in video_input]).float() / 255.0
    raise ValueError(f"Unsupported video input format: {type(video_input)}")


def _build_sample_tchw_from_processed(processed_visuals, grid_thw, temporal_patch_size: int):
    sample_tensors = []
    frame_cursor = 0
    for grid in grid_thw:
        frame_count = int(grid[0]) * temporal_patch_size
        sample_tensors.append(processed_visuals[frame_cursor : frame_cursor + frame_count])
        frame_cursor += frame_count
    return torch.cat(sample_tensors, dim=0)


def _normalize_grid_rows(grid_thw) -> List:
    if grid_thw is None:
        return []

    if isinstance(grid_thw, torch.Tensor):
        if grid_thw.ndim == 1:
            return [grid_thw]
        if grid_thw.ndim == 2:
            return [grid_thw[i] for i in range(grid_thw.shape[0])]
        raise ValueError(f"Unsupported grid_thw tensor shape: {tuple(grid_thw.shape)}")

    if isinstance(grid_thw, np.ndarray):
        if grid_thw.ndim == 1:
            return [grid_thw]
        if grid_thw.ndim == 2:
            return [grid_thw[i] for i in range(grid_thw.shape[0])]
        raise ValueError(f"Unsupported grid_thw array shape: {grid_thw.shape}")

    if isinstance(grid_thw, (list, tuple)):
        if len(grid_thw) == 0:
            return []
        first = grid_thw[0]
        if isinstance(first, (int, np.integer)) or (torch.is_tensor(first) and first.ndim == 0):
            return [grid_thw]
        return list(grid_thw)

    raise ValueError(f"Unsupported grid_thw format: {type(grid_thw)}")


def _split_processed_visuals_by_sample(processed_visuals, grid_thw, visual_counts, temporal_patch_size: int):
    if processed_visuals is None:
        return []

    grid_rows = _normalize_grid_rows(grid_thw)
    sample_tensors = []
    frame_cursor = 0
    grid_cursor = 0
    total_frames = processed_visuals.shape[0]

    for visual_count in visual_counts:
        sample_grid_rows = grid_rows[grid_cursor : grid_cursor + visual_count]
        if len(sample_grid_rows) != visual_count:
            raise ValueError(
                f"Mismatch between sample visual count ({visual_count}) and grid rows ({len(sample_grid_rows)})."
            )

        sample_tensors.append(
            _build_sample_tchw_from_processed(
                processed_visuals[frame_cursor:total_frames],
                sample_grid_rows,
                temporal_patch_size,
            )
        )
        frame_cursor += sum(int(grid[0]) * temporal_patch_size for grid in sample_grid_rows)
        grid_cursor += visual_count

    if frame_cursor != total_frames:
        raise ValueError(f"Processed frame count mismatch: consumed {frame_cursor}, available {total_frames}.")
    if grid_cursor != len(grid_rows):
        raise ValueError(f"Processed grid row mismatch: consumed {grid_cursor}, available {len(grid_rows)}.")

    return sample_tensors


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

            image_inputs = []
            video_inputs = []
            image_sample_counts = []
            video_sample_counts = []
            for message in messages:
                sample_image_inputs, sample_video_inputs = process_vision_info(message)
                if sample_video_inputs:
                    video_inputs.extend(sample_video_inputs)
                    video_sample_counts.append(len(sample_video_inputs))
                elif sample_image_inputs:
                    image_inputs.extend(sample_image_inputs)
                    image_sample_counts.append(len(sample_image_inputs))

            inputs = self.processor(
                text=text,
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                return_tensors="pt",
                padding=True,
                padding_side="left",
            )
            temporal_patch_size = int(getattr(self.processor.image_processor, "temporal_patch_size", 1))
            processed_images = inputs.pop("processed_images", None)
            processed_videos = inputs.pop("processed_videos", None)

            image_tchw = []
            video_tchw = []
            if image_sample_counts:
                if processed_images is not None:
                    image_tchw = _split_processed_visuals_by_sample(
                        processed_images,
                        inputs.get("image_grid_thw"),
                        image_sample_counts,
                        temporal_patch_size,
                    )
                else:
                    image_offset = 0
                    for image_count in image_sample_counts:
                        sample_image_inputs = image_inputs[image_offset : image_offset + image_count]
                        image_tchw.append(_convert_image_list_to_tchw(sample_image_inputs))
                        image_offset += image_count

            if video_sample_counts:
                if processed_videos is not None:
                    video_tchw = _split_processed_visuals_by_sample(
                        processed_videos,
                        inputs.get("video_grid_thw"),
                        video_sample_counts,
                        temporal_patch_size,
                    )
                else:
                    video_offset = 0
                    for video_count in video_sample_counts:
                        sample_video_inputs = video_inputs[video_offset : video_offset + video_count]
                        sample_video_tensors = [_convert_video_input_to_tchw(video_input) for video_input in sample_video_inputs]
                        video_tchw.append(torch.cat(sample_video_tensors, dim=0))
                        video_offset += video_count

            inputs.update(
                {
                    "image_tchw": image_tchw if image_tchw else None,
                    "video_tchw": video_tchw if video_tchw else None,
                }
            )

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
