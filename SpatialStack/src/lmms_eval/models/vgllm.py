import base64
import copy
import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None

try:
    from transformers import AutoModelForCausalLM
except ImportError:
    AutoModelForCausalLM = None

try:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
except ImportError:
    get_class_from_dynamic_module = None

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from qwen_vl.data.utils import load_and_preprocess_images


def _video_to_pil_frames(video_path: str, max_num_frames: int) -> list[Image.Image]:
    vr = decord.VideoReader(video_path)
    image_num = len(vr)
    if image_num < max_num_frames:
        frame_indices = np.arange(image_num)
    else:
        frame_indices = np.linspace(0, image_num - 1, max_num_frames).astype(int)
    return [Image.fromarray(vr[i].asnumpy()).convert("RGB") for i in frame_indices]


def _normalize_visuals(visual, max_num_frames: int) -> list[Image.Image]:
    if visual is None:
        return []
    if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")):
        return _video_to_pil_frames(visual, max_num_frames)
    if isinstance(visual, Image.Image):
        return [visual.convert("RGB")]
    if isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
        return [v.convert("RGB") for v in visual]
    raise NotImplementedError(f"Unsupported visual type: {type(visual)}")


def _encode_image_to_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def _load_model_class(pretrained: str, architecture_name: Optional[str]):
    if architecture_name and get_class_from_dynamic_module and os.path.isdir(pretrained):
        for candidate in sorted(Path(pretrained).glob("modeling*.py")):
            class_ref = f"{candidate.stem}.{architecture_name}"
            try:
                return get_class_from_dynamic_module(class_ref, pretrained)
            except Exception:
                continue

    for auto_cls in (AutoModelForImageTextToText, AutoModelForVision2Seq, AutoModelForCausalLM):
        if auto_cls is not None:
            return auto_cls

    raise RuntimeError(
        "Could not find a compatible Transformers auto model class for VG-LLM. "
        "Please install a Transformers build with Qwen3-VL support."
    )


@register_model("vgllm")
class VGLLM(lmms):
    def __init__(
        self,
        pretrained: str,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        use_flash_attention_2: Optional[bool] = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        max_length: Optional[int] = None,
        add_frame_index: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.add_frame_index = add_frame_index
        self.max_num_frames = max_num_frames

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

        self._config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
        architecture_name = None
        if getattr(self._config, "architectures", None):
            architecture_name = self._config.architectures[0]

        load_class = _load_model_class(pretrained, architecture_name)
        load_kwargs = {
            "config": self._config,
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        if use_flash_attention_2:
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            load_kwargs["torch_dtype"] = "auto"

        self._model = load_class.from_pretrained(pretrained, **load_kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            padding_side="left",
            trust_remote_code=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left", trust_remote_code=True)

        if max_length is not None:
            eval_logger.warning(f"Setting max_length to {max_length}")
            setattr(self.processor.tokenizer, "model_max_length", max_length)
            setattr(self._tokenizer, "model_max_length", max_length)

        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

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
            self._model = self.model.to(self.device).to(torch.bfloat16)

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
        return getattr(self._tokenizer, "model_max_length", None)

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
        raise NotImplementedError("Loglikelihood is not implemented for VG-LLM")

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
            gen_kwargs = all_gen_kwargs[0]

            sample_visuals = []
            for sample_id in doc_id:
                raw_visuals = doc_to_visual[0](self.task_dict[task][split][sample_id])
                if raw_visuals is None:
                    raw_visuals = []
                elif not isinstance(raw_visuals, (list, tuple)):
                    raw_visuals = [raw_visuals]
                normalized = []
                for raw_visual in raw_visuals:
                    normalized.extend(_normalize_visuals(raw_visual, self.max_num_frames))
                sample_visuals.append(normalized)

            messages = []
            image_inputs = []
            geometry_encoder_inputs = []
            patch_size = self.processor.image_processor.patch_size
            merge_size = self.processor.image_processor.merge_size

            for context, visuals in zip(contexts, sample_visuals):
                image_content = []
                geometry_images = []
                image_count = 0
                for image in visuals:
                    if self.add_frame_index:
                        image_content.append({"type": "text", "text": f"Frame-{image_count}: "})
                    image_content.append({"type": "image", "image": _encode_image_to_data_uri(image)})
                    geometry_images.append(copy.deepcopy(image))
                    image_count += 1

                message = [{"role": "system", "content": "You are a helpful assistant."}]
                user_content = image_content + [{"type": "text", "text": context}] if image_content else [{"type": "text", "text": context}]
                message.append({"role": "user", "content": user_content})
                messages.append(message)

                cur_geometry_encoder_inputs = []
                for image in geometry_images:
                    image_tensor = load_and_preprocess_images([image])[0]
                    cur_geometry_encoder_inputs.append(copy.deepcopy(image_tensor))
                    _, height, width = image_tensor.shape
                    if (width // patch_size) % merge_size > 0:
                        width = width - (width // patch_size) % merge_size * patch_size
                    if (height // patch_size) % merge_size > 0:
                        height = height - (height // patch_size) % merge_size * patch_size
                    image_inputs.append(image_tensor[:, :height, :width])

                if cur_geometry_encoder_inputs:
                    geometry_encoder_inputs.append(torch.stack(cur_geometry_encoder_inputs))

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(
                text=text,
                images=image_inputs if image_inputs else None,
                videos=None,
                padding=True,
                return_tensors="pt",
                do_rescale=False,
            )

            device = "cuda" if self.device_map == "auto" else self.device
            if geometry_encoder_inputs and (
                getattr(self.model.config, "use_geometry_encoder", False)
                or getattr(self.model.config, "use_vggt_feature", False)
            ):
                inputs["geometry_encoder_inputs"] = [feat.to(device) for feat in geometry_encoder_inputs]
            inputs = inputs.to(device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=bool(gen_kwargs["temperature"] > 0),
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)]
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
