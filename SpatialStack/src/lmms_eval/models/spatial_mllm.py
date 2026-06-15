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


def _split_processed_visuals_by_sample(
    processed_visuals,
    grid_thw,
    visual_counts,
    temporal_patch_size: int,
    patch_size: int,
):
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

        sample_tensor = _build_sample_tchw_from_processed(
            processed_visuals[frame_cursor:total_frames],
            sample_grid_rows,
            temporal_patch_size,
        )
        expected_h = int(sample_grid_rows[0][1]) * patch_size
        expected_w = int(sample_grid_rows[0][2]) * patch_size
        actual_h, actual_w = sample_tensor.shape[-2:]
        if (actual_h, actual_w) != (expected_h, expected_w):
            raise ValueError(
                "Processed visual size and grid_thw are inconsistent: "
                f"actual HxW={actual_h}x{actual_w}, expected HxW={expected_h}x{expected_w}."
            )
        sample_tensors.append(sample_tensor)
        frame_cursor += sum(int(grid[0]) * temporal_patch_size for grid in sample_grid_rows)
        grid_cursor += visual_count

    if frame_cursor != total_frames:
        raise ValueError(f"Processed frame count mismatch: consumed {frame_cursor}, available {total_frames}.")
    if grid_cursor != len(grid_rows):
        raise ValueError(f"Processed grid row mismatch: consumed {grid_cursor}, available {len(grid_rows)}.")

    return sample_tensors


def _left_pad_tensors(tensors: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_length = max(tensor.numel() for tensor in tensors)
    padded = tensors[0].new_full((len(tensors), max_length), pad_value)
    for idx, tensor in enumerate(tensors):
        padded[idx, -tensor.numel():] = tensor
    return padded


def _cat_present_tensors(sample_inputs: List[dict], key: str):
    tensors = [sample_input[key] for sample_input in sample_inputs if sample_input.get(key) is not None]
    if not tensors:
        return None
    return torch.cat(tensors, dim=0)


def _merge_second_per_grid_ts(sample_inputs: List[dict]):
    values = []
    for sample_input in sample_inputs:
        second_per_grid_ts = sample_input.get("second_per_grid_ts")
        if second_per_grid_ts is None:
            continue
        if torch.is_tensor(second_per_grid_ts):
            values.extend(second_per_grid_ts.tolist())
        else:
            values.extend(second_per_grid_ts)
    return values or None


def _collate_spatial_mllm_inputs(sample_inputs: List[dict], pad_token_id: int) -> dict:
    input_ids = _left_pad_tensors(
        [sample_input["input_ids"].view(-1) for sample_input in sample_inputs],
        pad_token_id,
    )
    attention_mask = _left_pad_tensors(
        [sample_input["attention_mask"].view(-1) for sample_input in sample_inputs],
        0,
    )

    image_tchw = []
    video_tchw = []
    for sample_input in sample_inputs:
        image_tchw.extend(sample_input.get("image_tchw") or [])
        video_tchw.extend(sample_input.get("video_tchw") or [])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": _cat_present_tensors(sample_inputs, "pixel_values"),
        "image_grid_thw": _cat_present_tensors(sample_inputs, "image_grid_thw"),
        "pixel_values_videos": _cat_present_tensors(sample_inputs, "pixel_values_videos"),
        "video_grid_thw": _cat_present_tensors(sample_inputs, "video_grid_thw"),
        "second_per_grid_ts": _merge_second_per_grid_ts(sample_inputs),
        "image_tchw": image_tchw or None,
        "video_tchw": video_tchw or None,
    }


def _move_inputs_to_device(inputs: dict, device) -> dict:
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            inputs[key] = value.to(device)
    if inputs.get("image_tchw") is not None:
        inputs["image_tchw"] = [image_tchw.to(device) for image_tchw in inputs["image_tchw"]]
    if inputs.get("video_tchw") is not None:
        inputs["video_tchw"] = [video_tchw.to(device) for video_tchw in inputs["video_tchw"]]
    return inputs


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
    image_processor_module = importlib.import_module("src.qwenvl.preprocessor.image_processing_qwen2_vl")
    return (
        spatial_mllm_module.SpatialMLLMConfig,
        spatial_mllm_module.SpatialMLLMForConditionalGeneration,
        image_processor_module.Qwen2VLImageProcessorModified,
    )


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

        (
            SpatialMLLMConfig,
            SpatialMLLMForConditionalGeneration,
            Qwen2VLImageProcessorModified,
        ) = _import_spatial_mllm_modules(spatial_mllm_repo_path)
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
        self.processor.image_processor = Qwen2VLImageProcessorModified.from_pretrained(
            pretrained,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
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
        self._spatial_patch_size = int(self.model.config.spatial_config.patch_size)
        processor_patch_size = int(self.processor.image_processor.patch_size)
        if processor_patch_size != self._spatial_patch_size:
            raise ValueError(
                "Spatial-MLLM processor and spatial encoder patch sizes differ: "
                f"processor={processor_patch_size}, spatial_encoder={self._spatial_patch_size}."
            )

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

            sample_inputs = []
            temporal_patch_size = int(getattr(self.processor.image_processor, "temporal_patch_size", 1))
            for message in messages:
                sample_text = self.processor.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                sample_image_inputs, sample_video_inputs = process_vision_info(message)

                sample_input = self.processor(
                    text=[sample_text],
                    images=sample_image_inputs if sample_image_inputs else None,
                    videos=sample_video_inputs if sample_video_inputs else None,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                )
                processed_images = sample_input.pop("processed_images", None)
                processed_videos = sample_input.pop("processed_videos", None)

                image_tchw = []
                video_tchw = []
                if sample_image_inputs:
                    if processed_images is None:
                        raise RuntimeError(
                            "Spatial-MLLM processor did not return `processed_images`; "
                            "the adapter requires Qwen2VLImageProcessorModified for spatial encoder alignment."
                        )
                    image_tchw = _split_processed_visuals_by_sample(
                        processed_images,
                        sample_input.get("image_grid_thw"),
                        [len(sample_image_inputs)],
                        temporal_patch_size,
                        self._spatial_patch_size,
                    )

                if sample_video_inputs:
                    if processed_videos is None:
                        raise RuntimeError(
                            "Spatial-MLLM processor did not return `processed_videos`; "
                            "the adapter requires Qwen2VLImageProcessorModified for spatial encoder alignment."
                        )
                    video_tchw = _split_processed_visuals_by_sample(
                        processed_videos,
                        sample_input.get("video_grid_thw"),
                        [len(sample_video_inputs)],
                        temporal_patch_size,
                        self._spatial_patch_size,
                    )

                sample_input.update(
                    {
                        "image_tchw": image_tchw if image_tchw else None,
                        "video_tchw": video_tchw if video_tchw else None,
                    }
                )
                sample_inputs.append(sample_input)

            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            inputs = _collate_spatial_mllm_inputs(sample_inputs, pad_token_id)
            inputs = {key: value for key, value in inputs.items() if value is not None}

            if inputs.get("pixel_values") is not None and inputs.get("image_tchw") is None:
                raise RuntimeError("`image_tchw` must be provided when `pixel_values` is present.")
            if inputs.get("pixel_values_videos") is not None and inputs.get("video_tchw") is None:
                raise RuntimeError("`video_tchw` must be provided when `pixel_values_videos` is present.")

            device = "cuda" if self.device_map == "auto" else self.device
            inputs = _move_inputs_to_device(inputs, device)

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
                pad_token_id=pad_token_id,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
            ]
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
