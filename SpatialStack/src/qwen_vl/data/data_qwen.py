import copy
import json
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import transformers
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers.video_utils import VideoMetadata

from . import data_list
from .utils import prepare_video_inputs


IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
QWEN_VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"

QWEN3_5_NON_THINKING_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' }}"
    "{% if message['role'] == 'assistant' %}{{ '<think>\n\n</think>\n\n' + message['content'] }}{% else %}{{ message['content'] }}{% endif %}"
    "{{ '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n\n</think>\n\n' }}{% endif %}"
)

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path, max_samples: int = -1):
    records = []
    with open(path, "r") as file:
        for line in file:
            records.append(json.loads(line))
            if max_samples != -1 and len(records) >= max_samples:
                break
    return records


def _build_training_tokenizer(tokenizer: transformers.PreTrainedTokenizer):
    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.chat_template = QWEN3_5_NON_THINKING_CHAT_TEMPLATE
    return tokenizer


def _apply_training_chat_template(tokenizer, messages, *, tokenize=True):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=False,
    )


def _get_assistant_prefix_length(tokenizer: transformers.PreTrainedTokenizer) -> int:
    empty_assistant_ids = _apply_training_chat_template(
        tokenizer,
        [{"role": "assistant", "content": ""}],
    )
    closing_ids = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    return max(0, len(empty_assistant_ids) - len(closing_ids))


def preprocess_video(
    source,
    tokenizer: transformers.PreTrainedTokenizer,
    processor,
    videos: List[torch.Tensor],
    video_metadata: List[VideoMetadata],
) -> Dict[str, torch.Tensor]:
    """Tokenize one conversation through the native Qwen3.5 video processor."""
    roles = {"human": "user", "gpt": "assistant"}
    tokenizer = _build_training_tokenizer(tokenizer)
    assistant_prefix_length = _get_assistant_prefix_length(tokenizer)
    if source:
        first_role = source[0].get("from", source[0].get("role"))
    else:
        first_role = None
    if source and roles.get(first_role, first_role) != "user":
        source = source[1:]

    input_ids = []
    labels = []
    mm_token_type_ids = []
    pixel_values_videos = []
    video_grid_thw = []
    video_offset = 0

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.extend(source)
    for message in messages:
        raw_role = message.get("from", message.get("role"))
        role = roles.get(raw_role, raw_role)
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported conversation role: {raw_role!r}.")
        content = message.get("value", message.get("content", ""))
        num_videos = content.count(DEFAULT_VIDEO_TOKEN)
        if role != "user" and num_videos:
            raise ValueError("SceneDistill visual placeholders must appear in user messages.")
        content = content.replace(DEFAULT_VIDEO_TOKEN, QWEN_VIDEO_PLACEHOLDER)
        rendered = _apply_training_chat_template(
            tokenizer,
            [{"role": role, "content": content}],
            tokenize=False,
        )

        processor_kwargs = {
            "text": [rendered],
            "images": None,
            "padding": False,
            "return_tensors": "pt",
            "return_mm_token_type_ids": True,
        }
        if num_videos:
            next_offset = video_offset + num_videos
            processor_kwargs.update(
                videos=videos[video_offset:next_offset],
                video_metadata=video_metadata[video_offset:next_offset],
                do_sample_frames=False,
                do_resize=False,
            )
            video_offset = next_offset
        else:
            processor_kwargs["videos"] = None

        encoded = processor(**processor_kwargs)
        encoded_ids = encoded["input_ids"][0]
        encoded_mm_types = encoded.get("mm_token_type_ids")
        if encoded_mm_types is None:
            raise RuntimeError("Qwen3.5 processor did not return mm_token_type_ids.")
        encoded_mm_types = encoded_mm_types[0]
        if encoded_ids.shape != encoded_mm_types.shape:
            raise ValueError(
                "Qwen3.5 token/type-id shape mismatch: "
                f"input_ids={tuple(encoded_ids.shape)}, mm_token_type_ids={tuple(encoded_mm_types.shape)}."
            )

        input_ids.append(encoded_ids)
        mm_token_type_ids.append(encoded_mm_types)
        if role in {"user", "system"}:
            labels.append(torch.full_like(encoded_ids, IGNORE_INDEX))
        else:
            target = encoded_ids.clone()
            target[: min(assistant_prefix_length, target.numel())] = IGNORE_INDEX
            labels.append(target)

        if num_videos:
            grids = encoded["video_grid_thw"]
            if grids.shape[0] != num_videos:
                raise ValueError(
                    f"Qwen processor returned {grids.shape[0]} video grids for {num_videos} placeholders."
                )
            pixel_values_videos.append(encoded["pixel_values_videos"])
            video_grid_thw.append(grids)

    if video_offset != len(videos) or len(videos) != len(video_metadata):
        raise ValueError(
            "SceneDistill video contract mismatch: "
            f"consumed={video_offset}, videos={len(videos)}, metadata={len(video_metadata)}."
        )

    result = {
        "input_ids": torch.cat(input_ids),
        "labels": torch.cat(labels),
        "mm_token_type_ids": torch.cat(mm_token_type_ids),
    }
    if pixel_values_videos:
        result["pixel_values_videos"] = torch.cat(pixel_values_videos, dim=0)
        result["video_grid_thw"] = torch.cat(video_grid_thw, dim=0)
    return result


def _estimated_video_groups(sample: dict, max_frames: int, temporal_patch_size: int) -> int:
    if "video" in sample:
        num_videos = len(sample["video"]) if isinstance(sample["video"], list) else 1
        return num_videos * ((max_frames + temporal_patch_size - 1) // temporal_patch_size)
    visual = sample.get("images", sample.get("image"))
    if visual is None:
        return 0
    num_frames = len(visual) if isinstance(visual, list) else 1
    num_frames = min(num_frames, max_frames)
    return (num_frames + temporal_patch_size - 1) // temporal_patch_size


class LazySupervisedDataset(Dataset):
    """SceneDistill SFT dataset with one native-video path for every visual input."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super().__init__()
        if (
            data_args.model_type != "qwen3.5"
            or getattr(data_args, "geometry_encoder_type", None) != "scene_distill"
            or getattr(data_args, "processor", None) is None
        ):
            raise ValueError("SceneDistill data loading requires the complete Qwen3.5 processor.")

        dataset_list = data_list(data_args.dataset_use.split(","))
        print(f"Loading datasets: {dataset_list}")
        records = []
        for data in dataset_list:
            if data["annotation_path"].endswith(".jsonl"):
                annotations = read_jsonl(data["annotation_path"], data_args.max_samples)
            else:
                with open(data["annotation_path"], "r") as file:
                    annotations = json.load(file)
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(annotations, int(len(annotations) * sampling_rate))
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for annotation in annotations:
                annotation["data_path"] = data["data_path"]
                annotation["tag"] = data["tag"]
            records.extend(annotations)

        print(f"Total training samples: {len(records)}")
        if data_args.shuffle:
            random.shuffle(records)
        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = records
        self.data_args = data_args
        self.processor = data_args.processor
        self.video_processor = self.processor.video_processor

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        max_frames = int(self.data_args.video_max_frames)
        temporal_patch_size = int(self.video_processor.temporal_patch_size)
        patch_factor = int(self.video_processor.patch_size) * int(
            self.video_processor.merge_size
        )
        max_frame_pixels = int(self.data_args.video_max_frame_pixels)
        visual_tokens_per_group = (
            max_frame_pixels + patch_factor**2 - 1
        ) // patch_factor**2
        tokens_per_group = visual_tokens_per_group + 17
        return [
            sum(
                len(conv.get("value", conv.get("content", "")).split())
                for conv in sample["conversations"]
            )
            + _estimated_video_groups(sample, max_frames, temporal_patch_size) * tokens_per_group
            for sample in self.list_data_dict
        ]

    @property
    def modality_lengths(self):
        lengths = self.lengths
        return [
            -length if sample.get("tag", "2d") == "2d" else length
            for length, sample in zip(lengths, self.list_data_dict)
        ]

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            return np.array([sample["num_tokens"] for sample in self.list_data_dict])
        print("No pre-calculated length available.")
        return np.ones(len(self.list_data_dict), dtype=np.int64)

    def draw_visual_marks(self, frames, spar_info):
        if spar_info is None:
            return
        info = json.loads(spar_info)
        from .draw_marker import DRAW_FUNCTIONS

        draw_fn = DRAW_FUNCTIONS[info["type"]]
        draw_fn(frames[0] if len(frames) == 1 else frames, info)

    def process_video(self, visual, data_path, spar_info=None):
        """Load and sample one logical video without duplicating real frames."""
        backend = "image_sequence"
        raw_fps = 1.0

        if isinstance(visual, (list, tuple)):
            if not visual:
                raise ValueError("SceneDistill received an empty image sequence.")
            frame_sources = [
                frame
                if not isinstance(frame, str) or os.path.isabs(frame)
                else os.path.join(data_path, frame)
                for frame in visual
            ]
        elif isinstance(visual, Image.Image):
            frame_sources = [visual]
        elif isinstance(visual, str):
            visual_path = visual if os.path.isabs(visual) else os.path.join(data_path, visual)
            if not os.path.exists(visual_path):
                raise FileNotFoundError(f"Visual input does not exist: {visual_path}")
            if os.path.isdir(visual_path):
                frame_sources = sorted(
                    os.path.join(visual_path, name)
                    for name in os.listdir(visual_path)
                    if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
                )
                if not frame_sources:
                    raise ValueError(f"Frame directory is empty: {visual_path}")
                backend = "frame_directory"
            elif visual_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                frame_sources = [visual_path]
            else:
                reader = VideoReader(visual_path, num_threads=4)
                total_frames = len(reader)
                raw_fps = float(reader.get_avg_fps())
                if total_frames <= 0 or raw_fps <= 0:
                    raise ValueError(
                        f"Invalid video metadata for {visual_path}: frames={total_frames}, fps={raw_fps}."
                    )
                duration = total_frames / raw_fps
                target_frames = round(duration / float(self.data_args.base_interval))
                target_frames = min(
                    max(target_frames, int(self.data_args.video_min_frames)),
                    int(self.data_args.video_max_frames),
                    total_frames,
                )
                indices = np.unique(
                    np.linspace(0, total_frames - 1, target_frames).round().astype(np.int64)
                )
                arrays = reader.get_batch(indices).asnumpy()
                frames = [Image.fromarray(array).convert("RGB") for array in arrays]
                backend = "decord"
                frame_sources = None

        if frame_sources is not None:
            total_frames = len(frame_sources)
            target_frames = min(total_frames, int(self.data_args.video_max_frames))
            indices = np.unique(
                np.linspace(0, total_frames - 1, target_frames).round().astype(np.int64)
            )
            selected = [frame_sources[int(index)] for index in indices]
            frames = [
                frame.convert("RGB") if isinstance(frame, Image.Image) else Image.open(frame).convert("RGB")
                for frame in selected
            ]

        self.draw_visual_marks(frames, spar_info)
        prepared = prepare_video_inputs(
            frames,
            self.video_processor,
            min_pixels=int(self.data_args.video_min_frame_pixels),
            max_pixels=int(self.data_args.video_max_frame_pixels),
        )
        video_frames = prepared["video_frames"]
        metadata = VideoMetadata(
            total_num_frames=int(total_frames),
            fps=float(raw_fps),
            width=int(video_frames.shape[-1]),
            height=int(video_frames.shape[-2]),
            duration=float(total_frames / raw_fps),
            video_backend=backend,
            frames_indices=[int(index) for index in indices],
        )
        return video_frames, metadata, prepared["geometry_encoder_inputs"]

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        return self._get_item(index)

    def _get_item(self, index) -> Dict[str, torch.Tensor]:
        sample = copy.deepcopy(self.list_data_dict[index])
        visual_fields = [name for name in ("image", "images", "video") if name in sample]
        if len(visual_fields) > 1:
            raise ValueError(f"A sample must use exactly one visual field, got {visual_fields}.")

        visual_objects = []
        if visual_fields:
            visual_field = visual_fields[0]
            visual_value = sample[visual_field]
            if visual_field == "video":
                visual_objects = list(visual_value) if isinstance(visual_value, list) else [visual_value]
            else:
                visual_objects = [visual_value]

        conversations = sample["conversations"]
        placeholder_count = sum(
            message.get("value", message.get("content", "")).count(DEFAULT_IMAGE_TOKEN)
            + message.get("value", message.get("content", "")).count(DEFAULT_VIDEO_TOKEN)
            for message in conversations
        )
        if visual_objects and placeholder_count == 0:
            for message in conversations:
                role = message.get("from", message.get("role"))
                if role in {"human", "user"}:
                    key = "value" if "value" in message else "content"
                    message[key] = DEFAULT_VIDEO_TOKEN * len(visual_objects) + "\n" + message[key]
                    break
        elif len(visual_objects) > 1 and placeholder_count == 1:
            for message in conversations:
                key = "value" if "value" in message else "content"
                content = message[key]
                if DEFAULT_IMAGE_TOKEN in content or DEFAULT_VIDEO_TOKEN in content:
                    token = DEFAULT_IMAGE_TOKEN if DEFAULT_IMAGE_TOKEN in content else DEFAULT_VIDEO_TOKEN
                    message[key] = content.replace(token, DEFAULT_VIDEO_TOKEN * len(visual_objects), 1)
                    break
        elif placeholder_count != len(visual_objects):
            raise ValueError(
                "SceneDistill placeholder/video mismatch: "
                f"placeholders={placeholder_count}, logical_videos={len(visual_objects)}."
            )

        for message in conversations:
            key = "value" if "value" in message else "content"
            message[key] = message[key].replace(DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN)
        final_placeholder_count = sum(
            message.get("value", message.get("content", "")).count(DEFAULT_VIDEO_TOKEN)
            for message in conversations
        )
        if final_placeholder_count != len(visual_objects):
            raise ValueError(
                "SceneDistill final placeholder/video mismatch: "
                f"placeholders={final_placeholder_count}, logical_videos={len(visual_objects)}."
            )

        videos = []
        metadata = []
        geometry_inputs = []
        for visual in visual_objects:
            video, video_metadata, geometry = self.process_video(
                visual,
                sample["data_path"],
                sample.get("spar_info"),
            )
            videos.append(video)
            metadata.append(video_metadata)
            geometry_inputs.append(geometry)

        data = preprocess_video(
            conversations,
            self.tokenizer,
            self.processor,
            videos,
            metadata,
        )
        if videos:
            data["geometry_encoder_inputs"] = geometry_inputs
        data["tag"] = sample.get("tag", "2d")
        return data


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer
    spatial_merge_size: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [instance["input_ids"] for instance in instances],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [instance["labels"] for instance in instances],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        mm_token_type_ids = torch.nn.utils.rnn.pad_sequence(
            [instance["mm_token_type_ids"] for instance in instances],
            batch_first=True,
            padding_value=0,
        )
        max_length = self.tokenizer.model_max_length
        input_ids = input_ids[:, :max_length]
        batch = {
            "input_ids": input_ids,
            "labels": labels[:, :max_length],
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "mm_token_type_ids": mm_token_type_ids[:, :max_length],
        }

        visual_instances = [instance for instance in instances if "pixel_values_videos" in instance]
        if visual_instances:
            batch["pixel_values_videos"] = torch.cat(
                [instance["pixel_values_videos"] for instance in visual_instances], dim=0
            )
            batch["video_grid_thw"] = torch.cat(
                [instance["video_grid_thw"] for instance in visual_instances], dim=0
            )
            batch["geometry_encoder_inputs"] = [
                video
                for instance in visual_instances
                for video in instance["geometry_encoder_inputs"]
            ]
            if batch["video_grid_thw"].shape[0] != len(batch["geometry_encoder_inputs"]):
                raise ValueError(
                    "SceneDistill collator lost video order: "
                    f"grids={batch['video_grid_thw'].shape[0]}, "
                    f"geometry_inputs={len(batch['geometry_encoder_inputs'])}."
                )
            expected_video_tokens = sum(
                int(t) * int(h) * int(w) // (self.spatial_merge_size**2)
                for t, h, w in batch["video_grid_thw"].tolist()
            )
            actual_video_tokens = int((batch["mm_token_type_ids"] == 2).sum().item())
            if actual_video_tokens != expected_video_tokens:
                raise ValueError(
                    "SceneDistill sequence truncation or packing mismatch: "
                    f"video_tokens={actual_video_tokens}, expected={expected_video_tokens}."
                )
            tags = [instance.get("tag", "3d") for instance in instances]
            if len(set(tags)) != 1:
                raise ValueError("All samples in a SceneDistill batch must share the same tag.")
            batch["tag"] = tags[0]
        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> Dict:
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": DataCollatorForSupervisedDataset(
            tokenizer=tokenizer,
            spatial_merge_size=int(data_args.processor.video_processor.merge_size),
        ),
    }
