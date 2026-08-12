import re
import time
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from packaging.version import Version
from PIL import Image
from tqdm import tqdm
import transformers
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from transformers.video_utils import VideoMetadata

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from qwen_vl.data.utils import load_and_preprocess_video_frames


MIN_QWEN3_5_TRANSFORMERS_VERSION = Version("5.3.0")


def require_qwen3_5_support():
    current_version = Version(transformers.__version__)
    if current_version < MIN_QWEN3_5_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Qwen3.5 evaluation requires transformers>="
            f"{MIN_QWEN3_5_TRANSFORMERS_VERSION}, but found {transformers.__version__}."
        )


def patch_qwen3_5_flash_attention():
    try:
        import transformers.modeling_flash_attention_utils as flash_attention_utils
    except ImportError:
        return

    if getattr(flash_attention_utils, "_spatialstack_qwen3_5_mrope_patch", False):
        return

    original_is_packed_sequence = flash_attention_utils._is_packed_sequence

    def patched_is_packed_sequence(position_ids, batch_size):
        if position_ids is not None and getattr(position_ids, "ndim", None) == 3:
            return False
        return original_is_packed_sequence(position_ids, batch_size)

    flash_attention_utils._is_packed_sequence = patched_is_packed_sequence
    flash_attention_utils._spatialstack_qwen3_5_mrope_patch = True


def is_image_path(path: str) -> bool:
    return path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def strip_thinking_content(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def detect_qwen3_5_fast_path_runtime():
    runtime = {}
    for module_name in ("fla", "causal_conv1d"):
        try:
            __import__(module_name)
            runtime[module_name] = True
        except ImportError:
            runtime[module_name] = False
    return runtime


def move_qwen3_5_eval_inputs_to_device(inputs, device):
    return inputs.to(device)


def parse_qwen3_5_layer_indices(value, name: str) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]

    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None

    parts = [part for part in re.split(r"[:;\s]+", text) if part]
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{name} must be a colon- or space-separated integer list, got {value!r}.") from exc


@register_model("qwen3_5")
class Qwen3_5(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3.5-4B",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        use_flash_attention_2: Optional[bool] = False,
        min_pixels: int = 12544,
        max_pixels: int = 262144,
        max_num_frames: int = 32,
        disable_thinking: bool = True,
        strip_thinking: bool = True,
        max_length: Optional[int] = None,
        geometry_encoder_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        require_qwen3_5_support()
        patch_qwen3_5_flash_attention()

        self.max_num_frames = max_num_frames
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.disable_thinking = disable_thinking
        self.strip_thinking = strip_thinking
        self.fast_path_runtime = detect_qwen3_5_fast_path_runtime()
        if not all(self.fast_path_runtime.values()):
            missing = ", ".join(name for name, available in self.fast_path_runtime.items() if not available)
            eval_logger.warning(
                f"Qwen3.5 optimized runtime dependencies are missing ({missing}). "
                "Upstream may fall back to slower torch kernels during eval."
            )

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

        config = AutoConfig.from_pretrained(pretrained)
        model_type = getattr(config, "model_type", None)
        if model_type not in {"qwen3_5", "qwen3_5_vl"}:
            raise ValueError(f"Unsupported model_type '{model_type}' for Qwen3.5 eval adapter.")
        prepare_config = getattr(self, "_prepare_config_for_eval", None)
        if prepare_config is not None:
            config, geometry_encoder_path = prepare_config(config, geometry_encoder_path)
        use_geometry_model = getattr(config, "use_geometry_encoder", False) or getattr(config, "use_vggt_feature", False)
        if use_geometry_model and int(batch_size) != 1:
            raise ValueError("Qwen3.5 geometry evaluation currently requires batch_size=1.")

        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Your transformers build does not expose Qwen3_5ForConditionalGeneration. "
                f"Please install transformers>={MIN_QWEN3_5_TRANSFORMERS_VERSION}."
            ) from exc

        geometry_encoder_path = geometry_encoder_path or getattr(config, "geometry_encoder_path", None)
        if use_geometry_model:
            geometry_encoder_type = getattr(config, "geometry_encoder_type", "vggt")
            if geometry_encoder_type == "scene_distill":
                from qwen_vl.model.modeling_qwen3_5_scene_distill import (
                    Qwen3_5ForConditionalGenerationWithSceneDistill,
                )

                load_class = Qwen3_5ForConditionalGenerationWithSceneDistill
            elif geometry_encoder_type == "vggt_omega_direct":
                from qwen_vl.model.modeling_qwen3_5_vggt_omega_direct import (
                    Qwen3_5ForConditionalGenerationWithVGGTOmegaDirect,
                )

                load_class = Qwen3_5ForConditionalGenerationWithVGGTOmegaDirect
            else:
                from qwen_vl.model.modeling_qwen3_5 import Qwen3_5ForConditionalGenerationWithGeometry

                load_class = Qwen3_5ForConditionalGenerationWithGeometry
        else:
            load_class = Qwen3_5ForConditionalGeneration

        load_kwargs = {
            "config": config,
            "torch_dtype": torch.bfloat16,
            "device_map": self.device_map,
        }
        if use_geometry_model and geometry_encoder_path:
            load_kwargs["geometry_encoder_path"] = geometry_encoder_path

        if use_flash_attention_2:
            self._model = load_class.from_pretrained(pretrained, attn_implementation="flash_attention_2", **load_kwargs).eval()
        else:
            self._model = load_class.from_pretrained(pretrained, **load_kwargs).eval()

        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            padding_side="left",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")
        if max_length is not None:
            setattr(self.processor.tokenizer, "model_max_length", max_length)
            setattr(self._tokenizer, "model_max_length", max_length)

        self._config = self.model.config
        self._max_length = getattr(self._tokenizer, "model_max_length", None)
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
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3.5")

    def _normalize_visual(self, visual_group):
        if isinstance(visual_group, tuple):
            visual_group = list(visual_group)
        if not isinstance(visual_group, list):
            return visual_group
        if len(visual_group) == 0:
            return None
        if len(visual_group) == 1:
            return visual_group[0]
        return visual_group

    def _sample_video_frames(self, video_path: str):
        vr = decord.VideoReader(video_path)
        frame_count = len(vr)
        fps = float(vr.get_avg_fps())
        if frame_count <= 0 or fps <= 0:
            raise ValueError(f"Invalid video metadata for {video_path}: frames={frame_count}, fps={fps}.")
        num_frames = min(frame_count, self.max_num_frames)
        indices = np.unique(
            np.linspace(0, frame_count - 1, num_frames).round().astype(np.int64)
        )
        frames = [Image.fromarray(vr[int(index)].asnumpy()).convert("RGB") for index in indices]
        metadata = VideoMetadata(
            total_num_frames=int(frame_count),
            fps=fps,
            duration=float(frame_count / fps),
            video_backend="decord",
            frames_indices=[int(index) for index in indices],
        )
        return frames, metadata

    def _build_sample(self, context, visual):
        sampled_videos = []
        video_metadata = []
        user_content = []

        if isinstance(visual, str) and visual.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
            visual_groups = [self._sample_video_frames(visual)]
        elif isinstance(visual, str) and is_image_path(visual):
            visual_groups = [([Image.open(visual).convert("RGB")], None)]
        elif isinstance(visual, Image.Image):
            visual_groups = [([visual.convert("RGB")], None)]
        elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
            visual_groups = [([frame.convert("RGB") for frame in visual], None)]
        elif isinstance(visual, (list, tuple)) and all(isinstance(v, str) and is_image_path(v) for v in visual):
            visual_groups = [([Image.open(path).convert("RGB") for path in visual], None)]
        elif isinstance(visual, (list, tuple)) and all(
            isinstance(v, str) and v.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
            for v in visual
        ):
            visual_groups = [self._sample_video_frames(path) for path in visual]
        elif visual is None:
            visual_groups = []
        else:
            raise TypeError(f"Unsupported Qwen3.5 visual input: {type(visual)}")

        for frames, metadata in visual_groups:
            total_frames = len(frames)
            if metadata is None:
                num_frames = min(total_frames, self.max_num_frames)
                indices = np.unique(
                    np.linspace(0, total_frames - 1, num_frames).round().astype(np.int64)
                )
                frames = [frames[int(index)] for index in indices]
                metadata = VideoMetadata(
                    total_num_frames=int(total_frames),
                    fps=1.0,
                    duration=float(total_frames),
                    video_backend="image_sequence",
                    frames_indices=[int(index) for index in indices],
                )
            video = load_and_preprocess_video_frames(
                frames,
                self.processor.video_processor,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            metadata.width = int(video.shape[-1])
            metadata.height = int(video.shape[-2])
            sampled_videos.append(video)
            video_metadata.append(metadata)
            user_content.append({"type": "video", "video": video})

        user_content.append({"type": "text", "text": context})
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ]
        return message, sampled_videos, video_metadata

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
            task_name = task[0]
            split_name = split[0]
            batched_visuals = [doc_to_visual[i](self.task_dict[task_name][split_name][ids]) for i, ids in enumerate(doc_id)]
            batch_start = time.perf_counter()

            gen_kwargs = dict(all_gen_kwargs[0])
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be Union[str, list], got {type(until)}")

            messages = []
            sample_videos = []
            sample_video_metadata = []
            for context, raw_visual in zip(contexts, batched_visuals):
                visual = self._normalize_visual(raw_visual)
                message, videos, metadata = self._build_sample(context, visual)
                messages.append(message)
                sample_videos.extend(videos)
                sample_video_metadata.extend(metadata)

            chat_template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self.disable_thinking:
                chat_template_kwargs["enable_thinking"] = False
            text = self.processor.apply_chat_template(messages, **chat_template_kwargs)
            text_batch = text if isinstance(text, list) else [text]
            inputs = self.processor(
                text=text_batch,
                images=None,
                videos=sample_videos or None,
                video_metadata=sample_video_metadata or None,
                do_sample_frames=False,
                do_resize=False,
                return_mm_token_type_ids=True,
                padding=True,
                return_tensors="pt",
            )
            preprocess_elapsed = time.perf_counter() - batch_start

            if self.device_map == "auto":
                inputs = move_qwen3_5_eval_inputs_to_device(inputs, "cuda")
            else:
                inputs = move_qwen3_5_eval_inputs_to_device(inputs, self.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            generate_start = time.perf_counter()
            output_ids = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=gen_kwargs["temperature"] > 0,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            generate_elapsed = time.perf_counter() - generate_start

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            decode_elapsed = time.perf_counter() - generate_start - generate_elapsed
            input_tokens = int(inputs.input_ids.shape[-1]) if hasattr(inputs, "input_ids") else -1
            output_tokens = (
                sum(int(ids.shape[-1]) for ids in generated_ids_trimmed) if generated_ids_trimmed else 0
            )
            eval_logger.debug(
                f"Qwen3.5 eval batch size={len(contexts)} input_tokens={input_tokens} "
                f"output_tokens={output_tokens} preprocess={preprocess_elapsed:.3f}s "
                f"generate={generate_elapsed:.3f}s decode={decode_elapsed:.3f}s "
                f"fast_path={self.fast_path_runtime}"
            )

            for answer, context in zip(answers, contexts):
                final_answer = strip_thinking_content(answer) if self.strip_thinking else answer
                res.append(final_answer)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), final_answer)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
