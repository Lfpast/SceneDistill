import base64
import copy
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from qwen_vl.data.utils import load_and_preprocess_images
from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT

try:
    from qwen_vl_utils import extract_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


def _decode_video_frames(video_path: str, max_num_frames: int) -> list[Image.Image]:
    vr = decord.VideoReader(video_path)
    image_num = len(vr)
    if image_num < max_num_frames:
        frame_indices = np.arange(image_num)
    else:
        frame_indices = np.linspace(0, image_num - 1, max_num_frames).astype(int)
    return [Image.fromarray(vr[i].asnumpy()).convert("RGB") for i in frame_indices]


def _normalize_sample_visuals(raw_visuals, max_num_frames: int) -> list[Image.Image]:
    if raw_visuals is None:
        return []
    if not isinstance(raw_visuals, (list, tuple)):
        raw_visuals = [raw_visuals]

    normalized = []
    for visual in raw_visuals:
        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")):
            normalized.extend(_decode_video_frames(visual, max_num_frames))
        elif isinstance(visual, Image.Image):
            normalized.append(visual.convert("RGB"))
        else:
            raise NotImplementedError(f"Unsupported visual type: {type(visual)}")
    return normalized


@register_model("vgllm")
class VGLLM(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        use_flash_attention_2: Optional[bool] = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,
        max_image_size: Optional[int] = None,
        max_length: Optional[int] = None,
        add_frame_index: bool = False,
        geometry_encoder_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.add_frame_index = add_frame_index
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        config = AutoConfig.from_pretrained(pretrained)
        model_type = getattr(config, "model_type", None)
        if model_type != "qwen2_5_vl":
            raise RuntimeError(
                f"VG-LLM source-tree adapter only supports Qwen2.5-VL checkpoints, but got model_type={model_type!r}. "
                "The public VG-LLM repository under /home/jackson/python/VG-LLM only ships "
                "`qwen_vl.model.modeling_qwen2_5_vl` and documents Qwen2.5-VL 3B/7B backbones in README.md. "
                "Your Qwen3-VL checkpoint therefore cannot be loaded from the current public source release."
            )

        if getattr(config, "use_geometry_encoder", False) or getattr(config, "use_vggt_feature", False):
            load_class = Qwen2_5_VLForConditionalGenerationWithVGGT
            eval_logger.info("Using Qwen2_5_VLForConditionalGenerationWithVGGT")
        else:
            load_class = Qwen2_5_VLForConditionalGeneration
            eval_logger.info("Using Qwen2_5_VLForConditionalGeneration")

        load_kwargs = {
            "config": config,
            "device_map": self.device_map,
        }
        if geometry_encoder_path is not None:
            load_kwargs["geometry_encoder_path"] = geometry_encoder_path
        if use_flash_attention_2:
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            load_kwargs["torch_dtype"] = "auto"
        self._model = load_class.from_pretrained(pretrained, **load_kwargs).eval()

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels, padding_side="left")
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")

        if max_length is not None:
            eval_logger.warning(f"Setting max_length to {max_length}")
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
            sample_visuals = [
                _normalize_sample_visuals(doc_to_visual[0](self.task_dict[task][split][ids]), self.max_num_frames)
                for ids in doc_id
            ]

            gen_kwargs = all_gen_kwargs[0]

            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            messages = []
            for visuals, context in zip(sample_visuals, contexts):
                message = [{"role": "system", "content": "You are a helpful assistant."}]
                if visuals:
                    image_content = []
                    image_count = 0
                    for image in visuals:
                        buffer = BytesIO()
                        image.convert("RGB").save(buffer, format="JPEG")
                        base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        if self.add_frame_index:
                            image_content.append({"type": "text", "text": f"Frame-{image_count}: "})
                        image_content.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"})
                        image_count += 1
                    message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
                else:
                    message.append({"role": "user", "content": [{"type": "text", "text": context}]})
                messages.append(message)

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            geometry_encoder_inputs = []
            image_inputs = []
            patch_size = self.processor.image_processor.patch_size
            merge_size = self.processor.image_processor.merge_size
            for message in messages:
                vision_info = extract_vision_info(message)
                cur_geometry_encoder_inputs = []
                for element in vision_info:
                    if "image" not in element:
                        raise NotImplementedError("Unsupported vision info type")
                    image = element["image"]
                    if isinstance(image, Image.Image):
                        pass
                    elif isinstance(image, str) and "base64," in image:
                        _, base64_data = image.split("base64,", 1)
                        data = base64.b64decode(base64_data)
                        with BytesIO(data) as bio:
                            image = copy.deepcopy(Image.open(bio))
                    else:
                        raise NotImplementedError("Unsupported image type")

                    image = load_and_preprocess_images([image])[0]
                    cur_geometry_encoder_inputs.append(copy.deepcopy(image))
                    _, height, width = image.shape
                    if (width // patch_size) % merge_size > 0:
                        width = width - (width // patch_size) % merge_size * patch_size
                    if (height // patch_size) % merge_size > 0:
                        height = height - (height // patch_size) % merge_size * patch_size
                    image_inputs.append(image[:, :height, :width])

                if cur_geometry_encoder_inputs:
                    geometry_encoder_inputs.append(torch.stack(cur_geometry_encoder_inputs))

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
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
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
