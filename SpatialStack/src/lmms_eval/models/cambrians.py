from __future__ import annotations

from io import BytesIO
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_pil

from cambrian.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from cambrian.conversation import conv_templates
from cambrian.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from cambrian.model.builder import load_pretrained_model

_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")


def _to_pil(image_like):
    if isinstance(image_like, Image.Image):
        return image_like.convert("RGB")
    if isinstance(image_like, str):
        return Image.open(image_like).convert("RGB")
    if isinstance(image_like, dict) and "bytes" in image_like:
        return Image.open(BytesIO(image_like["bytes"])).convert("RGB")
    raise TypeError(f"Unsupported visual type: {type(image_like)!r}")


def _is_video_path(visual_like) -> bool:
    return isinstance(visual_like, str) and visual_like.lower().endswith(_VIDEO_EXTENSIONS)


@register_model("cambrians")
class CambrianS(lmms):
    def __init__(
        self,
        pretrained: str = "",
        torch_dtype: Optional[Union[str, torch.dtype]] = "float16",
        batch_size: Optional[Union[int, str]] = 1,
        device_map: str = "cuda:0",
        conv_template: str = "qwen_2",
        use_cache: bool = True,
        truncate_context: bool = False,
        max_num_frames: Optional[int] = None,
        video_max_frames: int = 32,
        video_fps: int = 1,
        video_force_sample: bool = False,
        add_time_instruction: bool = False,
        disable_thinking: Optional[bool] = None,
        miv_token_len: int = 196,
        si_token_len: int = 729,
        image_aspect_ratio: str = "anyres",
        anyres_max_subimages: int = 9,
        use_flash_attention_2: Optional[bool] = False,
        max_length: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device_map if device_map != "auto" else "cuda:0")
            self.device_map = str(self._device)

        self.pretrained = pretrained
        self.model_name = get_model_name_from_path(pretrained) if pretrained else "cambrian-s"
        self.torch_dtype = torch_dtype
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        self.batch_size_per_gpu = int(batch_size)
        if max_num_frames is not None:
            video_max_frames = max_num_frames

        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
            pretrained,
            None,
            self.model_name,
            device_map=self.device_map,
            use_flash_attn=use_flash_attention_2,
        )
        self._model.config.video_max_frames = video_max_frames
        self._model.config.video_fps = video_fps
        self._model.config.video_force_sample = video_force_sample
        self._model.config.add_time_instruction = add_time_instruction
        self._model.config.miv_token_len = miv_token_len
        self._model.config.si_token_len = si_token_len
        self._model.config.image_aspect_ratio = image_aspect_ratio
        self._model.config.anyres_max_subimages = anyres_max_subimages

        if max_length is not None:
            self._max_length = max_length
            setattr(self._tokenizer, "model_max_length", max_length)

        self._config = self._model.config

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ]
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            self._rank = self.accelerator.process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
            self._model = self.model.to(self._device)

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
        raise NotImplementedError("Loglikelihood is not implemented for Cambrian-S")

    def _build_prompt(self, context: str, num_images: int) -> str:
        if num_images <= 0:
            return context
        if getattr(self._model.config, "mm_use_im_start_end", False):
            image_prefix = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        else:
            image_prefix = DEFAULT_IMAGE_TOKEN
        return f"{image_prefix * num_images}\n{context}"

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, requests, task_dict, tokenizer, image_processor, model_config, conv_template):
                self.requests = requests
                self.task_dict = task_dict
                self.tokenizer = tokenizer
                self.image_processor = image_processor
                self.model_config = model_config
                self.conv_template = conv_template

            def __len__(self):
                return len(self.requests)

            def __getitem__(self, idx):
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = self.requests[idx].args
                visuals = doc_to_visual(self.task_dict[task][split][doc_id])

                if visuals is None:
                    visuals = []
                elif not isinstance(visuals, (list, tuple)):
                    visuals = [visuals]

                normalized_visuals = []
                for visual in visuals:
                    if _is_video_path(visual):
                        normalized_visuals.extend(
                            read_video_pyav_pil(
                                visual,
                                num_frm=self.model_config.video_max_frames,
                                fps=self.model_config.video_fps,
                                force_include_last_frame=self.model_config.video_force_sample,
                            )
                        )
                    else:
                        normalized_visuals.append(_to_pil(visual))
                visuals = normalized_visuals

                if visuals:
                    if len(visuals) == 1:
                        visual_tensors, visual_sizes = process_images(visuals, self.image_processor, self.model_config)
                    else:
                        visual_tensors, visual_sizes = process_images(visuals, self.image_processor, self.model_config, use_pad=True)
                    if getattr(self.model_config, "mm_use_im_start_end", False):
                        image_prefix = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                    else:
                        image_prefix = DEFAULT_IMAGE_TOKEN
                    qs = f"{image_prefix * len(visuals)}\n{contexts}"
                else:
                    visual_tensors = None
                    visual_sizes = None
                    qs = contexts

                conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
                return input_ids, visual_tensors, visual_sizes, prompt, gen_kwargs

        dataset = Dataset(requests, self.task_dict, self.tokenizer, self._image_processor, self._config, self.conv_template)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0], num_workers=4, pin_memory=True)

        for _, (input_ids, visual_tensors, visual_sizes, cur_prompt, gen_kwargs) in enumerate(dataloader):
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 16
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            with torch.inference_mode():
                input_ids = input_ids.to(self.device)
                if visual_tensors is not None:
                    visual_tensors = [tensor.half().to(self.device) for tensor in visual_tensors]
                output_ids = self.model.generate(
                    inputs=input_ids,
                    images=visual_tensors,
                    image_sizes=visual_sizes,
                    use_cache=self.use_cache,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            eval_logger.debug(f"Question: {cur_prompt}")
            eval_logger.debug(f"Answer: {outputs}")
            res.append(outputs)
            pbar.update(1)

        return res
