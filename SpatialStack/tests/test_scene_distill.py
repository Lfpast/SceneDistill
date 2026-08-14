from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from qwen_vl.model.scene_distill_module import (
    LLM_BLOCK_INDICES,
    NUM_SPECIAL_TOKENS,
    POST_DISTILL_DEPTH,
    POST_DISTILL_WEIGHT,
    PRE_DISTILL_DEPTH,
    PRE_DISTILL_WEIGHT,
    PRE_VISION_BLOCK_INDICES,
    FrameCrossAttentionLayer,
    GlobalSelfAttentionLayer,
    SceneDistillPostModule,
    SceneDistillPreModule,
    scene_distillation_loss,
    select_pre_vision_layer_outputs,
)
from qwen_vl.model.vggt_omega_direct_config import (
    get_vggt_omega_direct_num_extra_tokens,
)
from qwen_vl.model.vggt_omega_direct_packing import (
    expand_image_embeds_with_direct_tokens,
    expand_visual_placeholders,
)

SCENE_DISTILL_PRE_STATE_PREFIX = "model.language_model.scene_distill.pre."
SCENE_DISTILL_POST_STATE_PREFIX = "model.language_model.scene_distill.post."


def _import_qwen35_module_or_skip(module_name):
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as error:
        pytest.skip(f"Qwen3.5 Transformers runtime is unavailable: {error}")


def test_special_token_initialization_uses_first_and_other_variants():
    module = SceneDistillPreModule(visual_dim=6, text_hidden_dim=10, stream_dim=8, num_heads=2)
    with torch.no_grad():
        module.camera[:, 0].fill_(1.0)
        module.camera[:, 1].fill_(2.0)
        module.scene[:, 0].fill_(3.0)
        module.scene[:, 1].fill_(4.0)

    tokens = module.prepare_pre_special_tokens([2, 3])

    assert tokens.shape == (5, NUM_SPECIAL_TOKENS, 8)
    torch.testing.assert_close(tokens[[0, 2], 0], torch.ones(2, 8))
    torch.testing.assert_close(tokens[[1, 3, 4], 0], torch.full((3, 8), 2.0))
    torch.testing.assert_close(tokens[[0, 2], 1:], torch.full((2, 16, 8), 3.0))
    torch.testing.assert_close(tokens[[1, 3, 4], 1:], torch.full((3, 16, 8), 4.0))


def test_frame_cross_attention_isolates_frames():
    torch.manual_seed(0)
    layer = FrameCrossAttentionLayer(
        special_dim=8, visual_dim=6, num_heads=2
    ).eval()
    special_tokens = torch.randn(2, NUM_SPECIAL_TOKENS, 8)
    visual_features = torch.randn(5, 6)

    baseline = layer(special_tokens, visual_features, frame_sizes=[2, 3])
    changed_features = visual_features.clone()
    changed_features[2:] *= -1.0
    changed = layer(special_tokens, changed_features, frame_sizes=[2, 3])

    torch.testing.assert_close(baseline[0], changed[0])
    assert not torch.allclose(baseline[1], changed[1])


def test_global_attention_isolates_videos():
    torch.manual_seed(1)
    layer = GlobalSelfAttentionLayer(
        special_dim=8, num_heads=2
    ).eval()
    special_tokens = torch.randn(3, NUM_SPECIAL_TOKENS, 8)

    baseline = layer(special_tokens, video_sizes=[2, 1])
    changed_tokens = special_tokens.clone()
    changed_tokens[2] += 10.0
    changed = layer(changed_tokens, video_sizes=[2, 1])

    torch.testing.assert_close(baseline[:2], changed[:2])
    assert not torch.allclose(baseline[2], changed[2])


def test_four_stage_module_shapes():
    torch.manual_seed(2)
    module = SceneDistillPreModule(visual_dim=6, text_hidden_dim=10, stream_dim=8, num_heads=2)
    visual_layers = [torch.randn(7, 6) for _ in PRE_VISION_BLOCK_INDICES]

    embeds, features, pre_global_tokens = module(
        visual_layers, frame_sizes=[2, 3, 2], video_sizes=[2, 1]
    )

    assert PRE_VISION_BLOCK_INDICES == (1, 5, 9, 13)
    assert PRE_DISTILL_DEPTH == 4
    assert features.shape == (3, NUM_SPECIAL_TOKENS, 16)
    assert embeds.shape == (3, NUM_SPECIAL_TOKENS, 10)
    assert pre_global_tokens.shape == (3, NUM_SPECIAL_TOKENS, 8)
    torch.testing.assert_close(features[..., 8:], pre_global_tokens)


def test_gcte_detaches_vision_and_trains_student_parameters():
    torch.manual_seed(4)
    module = SceneDistillPreModule(visual_dim=6, text_hidden_dim=10, stream_dim=8, num_heads=2)
    visual_layers = [torch.randn(4, 6, requires_grad=True) for _ in PRE_VISION_BLOCK_INDICES]

    embeds, features, _ = module(visual_layers, frame_sizes=[2, 2], video_sizes=[2])
    (embeds.sum() + features.sum()).backward()

    assert module.camera.grad is not None
    assert module.scene.grad is not None
    assert module.projector.fc2.weight.grad is not None
    assert all(visual_features.grad is None for visual_features in visual_layers)


def test_selects_exact_vision_block_outputs():
    hidden_states = [torch.full((1, 2), float(index)) for index in range(24)]

    selected = select_pre_vision_layer_outputs(hidden_states)

    assert [int(features[0, 0].item()) for features in selected] == [1, 5, 9, 13]


def test_distillation_loss_is_index_aligned_and_sums_tokens():
    teacher = torch.zeros(1, NUM_SPECIAL_TOKENS, 2048)
    for token_index in range(NUM_SPECIAL_TOKENS):
        teacher[0, token_index, token_index] = 1.0

    identical_loss = scene_distillation_loss(teacher.clone(), teacher)
    torch.testing.assert_close(identical_loss, torch.tensor(0.0))

    swapped = teacher.clone()
    swapped[:, [1, 2]] = swapped[:, [2, 1]]
    swapped_loss = scene_distillation_loss(swapped, teacher)
    torch.testing.assert_close(swapped_loss, torch.tensor(2.0))

    opposite = teacher.clone()
    opposite[:, 0] *= -1
    opposite_loss = scene_distillation_loss(opposite, teacher)
    torch.testing.assert_close(opposite_loss, torch.tensor(2.0))


def test_distillation_gradient_only_flows_to_student():
    student = torch.randn(2, NUM_SPECIAL_TOKENS, 2048, requires_grad=True)
    teacher = torch.randn(2, NUM_SPECIAL_TOKENS, 2048, requires_grad=True)

    distill_loss = scene_distillation_loss(student, teacher)
    sft_loss = student.sum() * 0.0 + 3.0
    total_loss = sft_loss + PRE_DISTILL_WEIGHT * distill_loss
    total_loss.backward()

    assert student.grad is not None
    assert teacher.grad is None
    torch.testing.assert_close(total_loss.detach(), 3.0 + PRE_DISTILL_WEIGHT * distill_loss.detach())


def test_special_tokens_are_prepended_per_frame():
    image_embeds = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    special_embeds = torch.stack(
        [torch.full((NUM_SPECIAL_TOKENS, 3), 100.0), torch.full((NUM_SPECIAL_TOKENS, 3), 200.0)]
    )

    expanded = expand_image_embeds_with_direct_tokens(
        image_embeds,
        special_embeds,
        patches_per_frame=[2, 3],
        insert_position="front",
    )

    torch.testing.assert_close(expanded[:NUM_SPECIAL_TOKENS], special_embeds[0])
    torch.testing.assert_close(expanded[NUM_SPECIAL_TOKENS:NUM_SPECIAL_TOKENS + 2], image_embeds[:2])
    second_frame_start = NUM_SPECIAL_TOKENS + 2
    torch.testing.assert_close(
        expanded[second_frame_start:second_frame_start + NUM_SPECIAL_TOKENS],
        special_embeds[1],
    )
    torch.testing.assert_close(expanded[-3:], image_embeds[-3:])


def test_direct_tokens_are_appended_per_temporal_group():
    video_embeds = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    direct_embeds = torch.stack(
        [torch.full((2, 3), 100.0), torch.full((2, 3), 200.0)]
    )

    expanded = expand_image_embeds_with_direct_tokens(
        video_embeds,
        direct_embeds,
        patches_per_frame=[2, 3],
        insert_position="back",
    )

    torch.testing.assert_close(expanded[:2], video_embeds[:2])
    torch.testing.assert_close(expanded[2:4], direct_embeds[0])
    torch.testing.assert_close(expanded[4:7], video_embeds[2:])
    torch.testing.assert_close(expanded[7:], direct_embeds[1])


@pytest.mark.parametrize("insert_position", ["front", "back"])
def test_placeholder_expansion_inserts_into_each_temporal_group(insert_position):
    input_ids = torch.tensor([[1, 99, 99, 7, 99, 99, 99, 2]])
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.tensor([[1, 4, 4, 5, 8, 8, 8, 9]])

    expanded_ids, expanded_labels, expanded_attention, expanded_positions = expand_visual_placeholders(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        position_ids=position_ids,
        placeholder_token_id=99,
        num_extra_per_frame=2,
        insert_position=insert_position,
    )

    assert expanded_ids.shape[1] == input_ids.shape[1] + 4
    assert torch.equal(expanded_attention, torch.ones_like(expanded_attention))
    inserted = [1, 2, 6, 7] if insert_position == "front" else [3, 4, 9, 10]
    assert torch.equal(expanded_labels[0, inserted], torch.full((4,), -100))
    assert expanded_positions[0, inserted[:2]].tolist() == [4, 4]
    assert expanded_positions[0, inserted[2:]].tolist() == [8, 8]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("camera", 1), ("scene16", 16), ("special17", 17)],
)
def test_direct_token_modes_have_expected_width(mode, expected):
    assert get_vggt_omega_direct_num_extra_tokens("vggt_omega_direct", mode) == expected


def test_direct_merged_frame_sizes_expand_each_video_temporal_group():
    direct_module = _import_qwen35_module_or_skip(
        "qwen_vl.model.modeling_qwen3_5_vggt_omega_direct"
    )

    sizes = direct_module.Qwen3_5ModelWithVGGTOmegaDirect._merged_frame_sizes(
        torch.tensor([[2, 4, 6], [1, 2, 2]]),
        spatial_merge_size=2,
    )

    assert sizes == [6, 6, 1]


def test_direct_temporal_alignment_keeps_video_boundaries():
    direct_module = _import_qwen35_module_or_skip(
        "qwen_vl.model.modeling_qwen3_5_vggt_omega_direct"
    )

    class FakeEncoder:
        def __init__(self, outputs):
            self.outputs = iter(outputs)

        def encode(self, _):
            return next(self.outputs)

    exact = torch.tensor([10.0, 20.0]).reshape(2, 1, 1)
    paired = torch.tensor([0.0, 2.0, 4.0, 6.0]).reshape(4, 1, 1)
    model = SimpleNamespace(geometry_encoder=FakeEncoder([exact, paired]))

    aligned = direct_module.Qwen3_5ModelWithVGGTOmegaDirect._collect_direct_features(
        model,
        [torch.zeros(2, 3, 1, 1), torch.zeros(4, 3, 1, 1)],
        torch.tensor([[2, 1, 1], [2, 1, 1]]),
        target_device=torch.device("cpu"),
        target_dtype=torch.float32,
    )

    torch.testing.assert_close(
        aligned[:, 0, 0],
        torch.tensor([10.0, 20.0, 1.0, 5.0]),
    )


def test_direct_temporal_alignment_uses_adaptive_pooling_for_non_two_to_one():
    direct_module = _import_qwen35_module_or_skip(
        "qwen_vl.model.modeling_qwen3_5_vggt_omega_direct"
    )

    class FakeEncoder:
        def encode(self, _):
            return torch.arange(5, dtype=torch.float32).reshape(5, 1, 1)

    model = SimpleNamespace(geometry_encoder=FakeEncoder())
    aligned = direct_module.Qwen3_5ModelWithVGGTOmegaDirect._collect_direct_features(
        model,
        [torch.zeros(5, 3, 1, 1)],
        torch.tensor([[2, 1, 1]]),
        target_device=torch.device("cpu"),
        target_dtype=torch.float32,
    )
    expected = torch.nn.functional.adaptive_avg_pool1d(
        torch.arange(5, dtype=torch.float32).reshape(1, 1, 5),
        2,
    ).reshape(2)

    torch.testing.assert_close(aligned[:, 0, 0], expected)


def test_direct_temporal_alignment_rejects_upsampling():
    direct_module = _import_qwen35_module_or_skip(
        "qwen_vl.model.modeling_qwen3_5_vggt_omega_direct"
    )

    class FakeEncoder:
        def encode(self, _):
            return torch.zeros(2, 1, 1)

    model = SimpleNamespace(geometry_encoder=FakeEncoder())
    with pytest.raises(ValueError, match="only 2 frames for 3 Qwen temporal groups"):
        direct_module.Qwen3_5ModelWithVGGTOmegaDirect._collect_direct_features(
            model,
            [torch.zeros(2, 3, 1, 1)],
            torch.tensor([[3, 1, 1]]),
            target_device=torch.device("cpu"),
            target_dtype=torch.float32,
        )


def test_placeholder_expansion_masks_prepend_labels():
    input_ids = torch.tensor([[1, 99, 99, 2]])
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)

    expanded_ids, expanded_labels, expanded_attention, _ = expand_visual_placeholders(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        position_ids=None,
        placeholder_token_id=99,
        num_extra_per_frame=NUM_SPECIAL_TOKENS,
        insert_position="front",
    )

    assert expanded_ids.shape[1] == input_ids.shape[1] + NUM_SPECIAL_TOKENS
    assert torch.equal(expanded_ids[0, 1:1 + NUM_SPECIAL_TOKENS], torch.full((NUM_SPECIAL_TOKENS,), 99))
    assert torch.equal(expanded_labels[0, 1:1 + NUM_SPECIAL_TOKENS], torch.full((NUM_SPECIAL_TOKENS,), -100))
    assert torch.equal(expanded_attention, torch.ones_like(expanded_attention))


def test_scene_distill_placeholder_positions_use_frame_center():
    input_ids = torch.tensor([[1, 99, 99, 99, 99, 2]])
    position_ids = torch.tensor(
        [
            [[7, 8, 8, 8, 8, 9]],
            [[7, 4, 4, 4, 4, 9]],
            [[7, 10, 10, 11, 11, 9]],
            [[7, 20, 21, 20, 21, 9]],
        ]
    )

    _, _, _, expanded_positions = expand_visual_placeholders(
        input_ids=input_ids,
        labels=None,
        attention_mask=torch.ones_like(input_ids),
        position_ids=position_ids,
        placeholder_token_id=99,
        num_extra_per_frame=NUM_SPECIAL_TOKENS,
        insert_position="front",
    )

    expected_anchor = torch.tensor([8, 4, 11, 21]).view(4, 1)
    torch.testing.assert_close(
        expanded_positions[:, 0, 1:1 + NUM_SPECIAL_TOKENS],
        expected_anchor.expand(-1, NUM_SPECIAL_TOKENS),
    )

def test_scene_distill_pre_module_state_dict_round_trip():
    torch.manual_seed(3)
    source = SceneDistillPreModule(visual_dim=6, text_hidden_dim=10, stream_dim=8, num_heads=2)
    target = SceneDistillPreModule(visual_dim=6, text_hidden_dim=10, stream_dim=8, num_heads=2)

    target.load_state_dict(source.state_dict(), strict=True)

    for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
        torch.testing.assert_close(source_parameter, target_parameter)


def test_scene_distill_checkpoint_ownership_matches_decoder_execution():
    root = nn.Module()
    root.model = nn.Module()
    root.model.language_model = nn.Module()
    root.model.language_model.scene_distill = nn.ModuleDict(
        {
            "pre": SceneDistillPreModule(
                visual_dim=6,
                text_hidden_dim=10,
                stream_dim=8,
                num_heads=2,
            ),
            "post": SceneDistillPostModule(
                llm_hidden_dim=10,
                special_dim=8,
                num_heads=2,
            ),
        }
    )

    keys = set(root.state_dict())
    pre_keys = {key for key in keys if key.startswith(SCENE_DISTILL_PRE_STATE_PREFIX)}
    post_keys = {key for key in keys if key.startswith(SCENE_DISTILL_POST_STATE_PREFIX)}

    assert len(pre_keys) == 176
    assert len(post_keys) == 258
    assert pre_keys | post_keys == {key for key in keys if "scene_distill" in key}
    assert f"{SCENE_DISTILL_PRE_STATE_PREFIX}camera" in pre_keys
    assert f"{SCENE_DISTILL_PRE_STATE_PREFIX}projector.fc1.weight" in pre_keys
    assert f"{SCENE_DISTILL_POST_STATE_PREFIX}inject.0.weight" in post_keys
    assert not any("_module" in key or "pre_" in key or "post_" in key for key in keys)


def test_scene_distill_checkpoint_loader_uses_exact_keys_only(tmp_path):
    modeling_qwen3_5 = _import_qwen35_module_or_skip(
        "qwen_vl.model.modeling_qwen3_5"
    )
    from safetensors.torch import save_file

    model = nn.Module()
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    model.model.language_model.scene_distill = nn.ModuleDict(
        {
            "pre": nn.Linear(2, 2, bias=False),
            "post": nn.Linear(2, 2, bias=False),
        }
    )

    pre_weight = torch.full((2, 2), 3.0)
    post_weight = torch.full((2, 2), 5.0)
    save_file(
        {
            f"{SCENE_DISTILL_PRE_STATE_PREFIX}weight": pre_weight,
            f"{SCENE_DISTILL_POST_STATE_PREFIX}weight": post_weight,
        },
        tmp_path / "model.safetensors",
    )

    loaded_keys = modeling_qwen3_5._load_qwen3_5_geometry_submodules(model, tmp_path)

    assert loaded_keys == {
        f"{SCENE_DISTILL_PRE_STATE_PREFIX}weight",
        f"{SCENE_DISTILL_POST_STATE_PREFIX}weight",
    }
    torch.testing.assert_close(model.model.language_model.scene_distill["pre"].weight, pre_weight)
    torch.testing.assert_close(
        model.model.language_model.scene_distill["post"].weight,
        post_weight,
    )


def test_checkpoint_filter_removes_external_geometry_but_keeps_learned_modules():
    modeling_qwen3_5 = _import_qwen35_module_or_skip(
        "qwen_vl.model.modeling_qwen3_5"
    )
    state_dict = {
        "model.geometry_encoder.vggt_omega.weight": torch.ones(1),
        "geometry_encoder.vggt_omega.weight": torch.ones(1),
        "model.direct_projector.fc1.weight": torch.ones(1),
        f"{SCENE_DISTILL_PRE_STATE_PREFIX}camera": torch.ones(1),
        f"{SCENE_DISTILL_POST_STATE_PREFIX}frame.0.q_proj.weight": torch.ones(1),
        f"{SCENE_DISTILL_POST_STATE_PREFIX}inject.0.weight": torch.ones(1),
        "model.language_model.weight": torch.ones(1),
    }

    filtered = modeling_qwen3_5.remove_geometry_encoder_weights(state_dict)

    assert set(filtered) == {
        "model.direct_projector.fc1.weight",
        f"{SCENE_DISTILL_PRE_STATE_PREFIX}camera",
        f"{SCENE_DISTILL_POST_STATE_PREFIX}frame.0.q_proj.weight",
        f"{SCENE_DISTILL_POST_STATE_PREFIX}inject.0.weight",
        "model.language_model.weight",
    }


def test_post_module_shapes_and_gradients():
    torch.manual_seed(5)
    module = SceneDistillPostModule(
        llm_hidden_dim=6,
        special_dim=8,
        num_heads=2,
        depth=POST_DISTILL_DEPTH,
    )
    pre_global_tokens = torch.randn(3, NUM_SPECIAL_TOKENS, 8, requires_grad=True)
    frame_sizes = [19, 20, 19]
    llm_layer_features = [
        torch.randn(sum(frame_sizes), 6, requires_grad=True)
        for _ in LLM_BLOCK_INDICES
    ]

    post_tokens = pre_global_tokens
    injection_deltas = []
    for stage_index, layer_features in enumerate(llm_layer_features):
        post_after_frame, post_tokens, injection_delta = module(
            stage_index,
            post_tokens,
            layer_features,
            frame_sizes=frame_sizes,
            video_sizes=[2, 1],
        )
        injection_deltas.append(injection_delta)
    post_features = torch.cat([post_after_frame, post_tokens], dim=-1)

    assert LLM_BLOCK_INDICES == (4, 8, 12, 16, 20, 24)
    assert POST_DISTILL_DEPTH == 6
    assert post_features.shape == (3, NUM_SPECIAL_TOKENS, 16)
    assert all(delta.shape == (3, NUM_SPECIAL_TOKENS, 6) for delta in injection_deltas)
    assert all(torch.count_nonzero(delta) == 0 for delta in injection_deltas)

    post_features.square().mean().backward()
    assert pre_global_tokens.grad is not None
    assert all(features.grad is not None for features in llm_layer_features)
    assert module.frame[0].q_proj.weight.grad is not None
    assert module.get_submodule("global")[-1].qkv.weight.grad is not None


def test_zero_gate_receives_sft_gradient_and_opens_post_gradient_path():
    torch.manual_seed(10)
    module = SceneDistillPostModule(llm_hidden_dim=6, special_dim=8, num_heads=2)
    post_tokens = torch.randn(1, NUM_SPECIAL_TOKENS, 8, requires_grad=True)
    llm_features = torch.randn(18, 6, requires_grad=True)

    injection_delta = module(0, post_tokens, llm_features, [18], [1])[2]
    injection_delta.sum().backward()

    assert module.inject[0].weight.grad.abs().sum() > 0
    assert module.frame[0].q_proj.weight.grad.abs().sum() == 0
    assert post_tokens.grad.abs().sum() == 0

    module.zero_grad(set_to_none=True)
    post_tokens.grad = None
    llm_features.grad = None
    with torch.no_grad():
        module.inject[0].weight.fill_(0.1)
    module(0, post_tokens, llm_features, [18], [1])[2].sum().backward()

    assert module.frame[0].q_proj.weight.grad.abs().sum() > 0
    assert post_tokens.grad.abs().sum() > 0


def test_post_frame_attention_uses_only_the_corresponding_frame_span():
    torch.manual_seed(6)
    module = SceneDistillPostModule(
        llm_hidden_dim=6,
        special_dim=8,
        num_heads=2,
    ).eval()
    pre_global_tokens = torch.randn(2, NUM_SPECIAL_TOKENS, 8)
    frame_sizes = [18, 20]
    llm_layer_features = [
        torch.randn(sum(frame_sizes), 6)
        for _ in LLM_BLOCK_INDICES
    ]

    baseline = module(
        0,
        pre_global_tokens,
        llm_layer_features[0],
        frame_sizes=frame_sizes,
        video_sizes=[1, 1],
    )[0]
    changed_features = llm_layer_features[0].clone()
    changed_features[frame_sizes[0]:] *= -2.0
    changed = module(
        0,
        pre_global_tokens,
        changed_features,
        frame_sizes=frame_sizes,
        video_sizes=[1, 1],
    )[0]

    torch.testing.assert_close(baseline[0], changed[0])
    assert not torch.allclose(baseline[1], changed[1])


def test_post_global_attention_isolates_videos():
    torch.manual_seed(7)
    layer = GlobalSelfAttentionLayer(
        special_dim=8,
        num_heads=2,
    ).eval()
    special_tokens = torch.randn(3, NUM_SPECIAL_TOKENS, 8)

    baseline = layer(special_tokens, video_sizes=[2, 1])
    changed_tokens = special_tokens.clone()
    changed_tokens[2] += 10.0
    changed = layer(changed_tokens, video_sizes=[2, 1])

    torch.testing.assert_close(baseline[:2], changed[:2])
    assert not torch.allclose(baseline[2], changed[2])


def test_post_global_attention_connects_frames_within_one_video():
    torch.manual_seed(9)
    layer = GlobalSelfAttentionLayer(
        special_dim=8,
        num_heads=2,
    ).eval()
    special_tokens = torch.randn(3, NUM_SPECIAL_TOKENS, 8)

    baseline = layer(special_tokens, video_sizes=[2, 1])
    changed_tokens = special_tokens.clone()
    changed_tokens[1] *= -2.0
    changed = layer(changed_tokens, video_sizes=[2, 1])

    assert not torch.allclose(baseline[0], changed[0])
    torch.testing.assert_close(baseline[2], changed[2])


def test_pre_and_post_modules_use_public_attention_classes_with_disjoint_parameters():
    pre_module = SceneDistillPreModule(
        visual_dim=6,
        text_hidden_dim=10,
        stream_dim=8,
        num_heads=2,
    )
    post_module = SceneDistillPostModule(
        llm_hidden_dim=10,
        special_dim=8,
        num_heads=2,
    )

    assert {id(parameter) for parameter in pre_module.parameters()}.isdisjoint(
        {id(parameter) for parameter in post_module.parameters()}
    )
    assert len({id(layer) for layer in post_module.frame}) == POST_DISTILL_DEPTH
    assert len({id(layer) for layer in post_module.get_submodule("global")}) == POST_DISTILL_DEPTH
    assert all(
        isinstance(layer, FrameCrossAttentionLayer)
        for layer in post_module.frame
    )
    assert all(
        isinstance(layer, GlobalSelfAttentionLayer)
        for layer in pre_module.get_submodule("global")
    )
    assert all(
        isinstance(layer, GlobalSelfAttentionLayer)
        for layer in post_module.get_submodule("global")
    )
    assert not any(
        component in name
        for name, _ in post_module.named_parameters()
        for component in ("camera_token", "scene_token", "projector")
    )
    assert len({id(projection.weight) for projection in post_module.inject}) == POST_DISTILL_DEPTH
    assert all(projection.bias is None for projection in post_module.inject)
    assert all(projection.weight.shape == (10, 8) for projection in post_module.inject)
    assert all(torch.count_nonzero(projection.weight) == 0 for projection in post_module.inject)
    with torch.no_grad():
        post_module.inject[0].weight.fill_(1)
    post_module.reset_parameters()
    assert all(torch.count_nonzero(projection.weight) == 0 for projection in post_module.inject)
    assert {key.split(".", 1)[0] for key in pre_module.state_dict()} == {
        "camera",
        "scene",
        "frame",
        "global",
        "projector",
    }


def test_post_module_rejects_invalid_stage_and_shapes():
    module = SceneDistillPostModule(
        llm_hidden_dim=6,
        special_dim=8,
        num_heads=2,
    )
    pre_global_tokens = torch.randn(2, NUM_SPECIAL_TOKENS, 8)
    frame_sizes = [18, 19]
    valid_features = torch.randn(sum(frame_sizes), 6)

    with pytest.raises(ValueError, match="stage_index"):
        module(
            POST_DISTILL_DEPTH,
            pre_global_tokens,
            valid_features,
            frame_sizes=frame_sizes,
            video_sizes=[2],
        )
    with pytest.raises(ValueError, match="features must have shape"):
        module(
            0,
            pre_global_tokens,
            torch.randn(sum(frame_sizes), 7),
            frame_sizes=frame_sizes,
            video_sizes=[2],
        )
    with pytest.raises(ValueError, match="post_tokens must have shape"):
        module(
            0,
            pre_global_tokens[:, :-1],
            valid_features,
            frame_sizes=frame_sizes,
            video_sizes=[2],
        )


def test_scene_distillation_loss_rejects_non_finite_inputs():
    teacher = torch.randn(1, NUM_SPECIAL_TOKENS, 2048)
    student = teacher.clone()
    student[0, 0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="student"):
        scene_distillation_loss(student, teacher)

    student = teacher.clone()
    teacher[0, 0, 0] = torch.inf
    with pytest.raises(FloatingPointError, match="teacher"):
        scene_distillation_loss(student, teacher)


def test_independent_pre_and_post_loss_weights():
    sft_loss = torch.tensor(3.0)
    pre_loss = torch.tensor(2.0)
    post_loss = torch.tensor(5.0)

    torch.testing.assert_close(
        sft_loss + 0.0 * pre_loss + POST_DISTILL_WEIGHT * post_loss,
        torch.tensor(3.0 + POST_DISTILL_WEIGHT * 5.0),
    )
    torch.testing.assert_close(
        sft_loss + PRE_DISTILL_WEIGHT * pre_loss + 0.0 * post_loss,
        torch.tensor(3.0 + PRE_DISTILL_WEIGHT * 2.0),
    )
    torch.testing.assert_close(
        sft_loss + PRE_DISTILL_WEIGHT * pre_loss + POST_DISTILL_WEIGHT * post_loss,
        torch.tensor(3.0 + PRE_DISTILL_WEIGHT * 2.0 + POST_DISTILL_WEIGHT * 5.0),
    )


def test_scene_distill_post_module_state_dict_round_trip():
    torch.manual_seed(8)
    source = SceneDistillPostModule(
        llm_hidden_dim=6,
        special_dim=8,
        num_heads=2,
    )
    target = SceneDistillPostModule(
        llm_hidden_dim=6,
        special_dim=8,
        num_heads=2,
    )
    with torch.no_grad():
        for index, projection in enumerate(source.inject, start=1):
            projection.weight.fill_(index)

    target.load_state_dict(source.state_dict(), strict=True)

    for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
        torch.testing.assert_close(source_parameter, target_parameter)
    assert all(torch.count_nonzero(projection.weight) > 0 for projection in target.inject)


def test_post_module_strict_load_rejects_missing_injection_weights():
    source = SceneDistillPostModule(llm_hidden_dim=6, special_dim=8, num_heads=2)
    stage2_state_dict = {
        key: value
        for key, value in source.state_dict().items()
        if not key.startswith("inject")
    }
    target = SceneDistillPostModule(llm_hidden_dim=6, special_dim=8, num_heads=2)
    with pytest.raises(RuntimeError, match="Missing key"):
        target.load_state_dict(stage2_state_dict, strict=True)
    assert all(torch.count_nonzero(projection.weight) == 0 for projection in target.inject)


def test_online_post_injection_uses_block_outputs_and_only_updates_special_tokens(monkeypatch):
    modeling_qwen3_5 = _import_qwen35_module_or_skip(
        "qwen_vl.model.modeling_qwen3_5"
    )
    assert "scene_distill" in modeling_qwen3_5.GEOMETRY_STATE_KEYWORDS

    class DummyDecoderLayer(nn.Module):
        layer_type = "full_attention"

        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, hidden_states, **kwargs):
            hidden_states = hidden_states + self.value
            if kwargs["past_key_values"] is not None:
                kwargs["past_key_values"].append(hidden_states.clone())
            return hidden_states

    class DummyRotaryEmbedding(nn.Module):
        def forward(self, hidden_states, position_ids):
            return None

    class ScaleNorm(nn.Module):
        def forward(self, hidden_states):
            return hidden_states * 10

    class FakePostModule(nn.Module):
        layer_indices = LLM_BLOCK_INDICES

        def __init__(self):
            super().__init__()
            self.calls = []
            self.injection_value = 0.0

        def forward(self, stage_index, post_tokens, llm_layer_features, frame_sizes, video_sizes):
            self.calls.append((LLM_BLOCK_INDICES[stage_index], llm_layer_features.clone()))
            next_tokens = post_tokens + 1
            injection_delta = torch.full(
                (1, NUM_SPECIAL_TOKENS, 8),
                self.injection_value,
                device=llm_layer_features.device,
                dtype=llm_layer_features.dtype,
            )
            return next_tokens - 0.5, next_tokens, injection_delta

    text_model = object.__new__(modeling_qwen3_5.Qwen3_5TextModelWithGeometry)
    nn.Module.__init__(text_model)
    text_model.config = SimpleNamespace(num_hidden_layers=25)
    text_model.layers = nn.ModuleList(
        DummyDecoderLayer(layer_index + 1)
        for layer_index in range(text_model.config.num_hidden_layers)
    )
    text_model.rotary_emb = DummyRotaryEmbedding()
    text_model.norm = ScaleNorm()
    linear_attn_cache_args = []
    text_model._update_linear_attn_mask = (
        lambda attention_mask, past_key_values: linear_attn_cache_args.append(past_key_values)
    )
    monkeypatch.setattr(modeling_qwen3_5, "create_causal_mask", lambda **kwargs: None)

    inputs_embeds = torch.zeros(1, 20, 8, requires_grad=True)
    image_mask = torch.zeros(1, 20, dtype=torch.bool)
    image_mask[:, :18] = True
    special_mask = torch.zeros(1, 20, dtype=torch.bool)
    special_mask[:, :NUM_SPECIAL_TOKENS] = True
    post_module = FakePostModule()
    baseline_cache = []
    baseline = text_model(
        inputs_embeds=inputs_embeds,
        use_cache=True,
        past_key_values=baseline_cache,
        cache_position=torch.arange(20),
    )
    text_model.scene_distill = nn.ModuleDict({"post": post_module})
    zero_gate_cache = []
    zero_gate_outputs = text_model(
        inputs_embeds=inputs_embeds,
        use_cache=True,
        past_key_values=zero_gate_cache,
        cache_position=torch.arange(20),
        scene_distill_post_tokens=torch.zeros(1, NUM_SPECIAL_TOKENS, 8),
        scene_distill_image_mask=image_mask,
        scene_distill_special_mask=special_mask,
        scene_distill_frame_sizes=[18],
        scene_distill_video_sizes=[1],
        return_scene_distill_post_features=True,
    )
    assert linear_attn_cache_args[0] is baseline_cache
    assert linear_attn_cache_args[1] is zero_gate_cache
    torch.testing.assert_close(zero_gate_outputs.last_hidden_state, baseline.last_hidden_state)
    assert len(zero_gate_cache) == len(baseline_cache) == 25
    for zero_gate_state, baseline_state in zip(zero_gate_cache, baseline_cache):
        torch.testing.assert_close(zero_gate_state, baseline_state)
    assert len(post_module.calls) == POST_DISTILL_DEPTH

    post_module.calls.clear()
    post_module.injection_value = 1.0
    outputs = text_model(
        inputs_embeds=inputs_embeds,
        use_cache=False,
        cache_position=torch.arange(20),
        scene_distill_post_tokens=torch.zeros(1, NUM_SPECIAL_TOKENS, 8),
        scene_distill_image_mask=image_mask,
        scene_distill_special_mask=special_mask,
        scene_distill_frame_sizes=[18],
        scene_distill_video_sizes=[1],
        return_scene_distill_post_features=True,
    )
    assert linear_attn_cache_args[2] is None

    assert [layer_index for layer_index, _ in post_module.calls] == list(LLM_BLOCK_INDICES)
    expected_special_outputs = [15, 46, 93, 156, 235, 330]
    expected_other_outputs = [15, 45, 91, 153, 231, 325]
    for (_, features), special_value, other_value in zip(
        post_module.calls,
        expected_special_outputs,
        expected_other_outputs,
    ):
        torch.testing.assert_close(
            features[:NUM_SPECIAL_TOKENS],
            torch.full_like(features[:NUM_SPECIAL_TOKENS], special_value),
        )
        torch.testing.assert_close(
            features[NUM_SPECIAL_TOKENS:],
            torch.full_like(features[NUM_SPECIAL_TOKENS:], other_value),
        )
    torch.testing.assert_close(
        outputs.last_hidden_state[0, :NUM_SPECIAL_TOKENS],
        torch.full((NUM_SPECIAL_TOKENS, 8), 3310.0),
    )
    torch.testing.assert_close(
        outputs.last_hidden_state[0, NUM_SPECIAL_TOKENS:],
        torch.full((20 - NUM_SPECIAL_TOKENS, 8), 3250.0),
    )
    torch.testing.assert_close(outputs.hidden_states[0], torch.full((1, NUM_SPECIAL_TOKENS, 8), 5.5))
    torch.testing.assert_close(outputs.hidden_states[1], torch.full((1, NUM_SPECIAL_TOKENS, 8), 6.0))

    outputs.last_hidden_state.sum().backward()
    assert inputs_embeds.grad is not None

    with pytest.raises(ValueError, match="provided together"):
        text_model(
            inputs_embeds=torch.zeros(1, 20, 8),
            use_cache=False,
            cache_position=torch.arange(20),
            scene_distill_post_tokens=torch.zeros(1, NUM_SPECIAL_TOKENS, 8),
        )


def test_online_post_injection_survives_gradient_checkpointing():
    modeling_qwen3_5 = _import_qwen35_module_or_skip(
        "qwen_vl.model.modeling_qwen3_5"
    )
    configuration_qwen3_5 = _import_qwen35_module_or_skip(
        "transformers.models.qwen3_5.configuration_qwen3_5"
    )
    config = configuration_qwen3_5.Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=25,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["full_attention"] * 25,
        use_cache=False,
    )
    model = modeling_qwen3_5.Qwen3_5TextModelWithGeometry(config).train()
    model.gradient_checkpointing_enable()
    input_ids = torch.randint(0, config.vocab_size, (1, 20))
    image_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    image_mask[:, :18] = True
    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    special_mask[:, :NUM_SPECIAL_TOKENS] = True
    post_module = SceneDistillPostModule(
        llm_hidden_dim=config.hidden_size,
        special_dim=8,
        num_heads=2,
    )
    with torch.no_grad():
        post_module.inject[0].weight.fill_(0.1)
    model.scene_distill = nn.ModuleDict({"post": post_module})

    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        scene_distill_post_tokens=torch.randn(1, NUM_SPECIAL_TOKENS, 8),
        scene_distill_image_mask=image_mask,
        scene_distill_special_mask=special_mask,
        scene_distill_frame_sizes=[18],
        scene_distill_video_sizes=[1],
        return_scene_distill_post_features=True,
    )
    loss = outputs.last_hidden_state.square().mean() + sum(
        features.square().mean() for features in outputs.hidden_states
    )
    loss.backward()

    assert model.is_gradient_checkpointing
    assert all(features.shape == (1, NUM_SPECIAL_TOKENS, 8) for features in outputs.hidden_states)
    assert model.layers[4].self_attn.q_proj.weight.grad is not None
    assert model.layers[24].self_attn.q_proj.weight.grad is not None


def test_scene_wrapper_validates_weights_layers_and_expanded_spans():
    scene_wrapper = _import_qwen35_module_or_skip(
        "qwen_vl.model.modeling_qwen3_5_scene_distill"
    )
    model_class = scene_wrapper.Qwen3_5ModelWithSceneDistill
    valid_config = SimpleNamespace(
        geometry_encoder_type="scene_distill",
        geometry_token_insert_position="front",
        geometry_direct_token_mode="special17",
        reference_frame="first",
        geometry_encoder_freeze=True,
        pre_distill_weight=0.05,
        post_distill_weight=0.05,
        text_config=SimpleNamespace(num_hidden_layers=25),
    )

    model_class._validate_geometry_config(None, valid_config)
    valid_config.pre_distill_weight = -0.1
    with pytest.raises(ValueError, match="pre_distill_weight"):
        model_class._validate_geometry_config(None, valid_config)
    valid_config.pre_distill_weight = 0.05
    valid_config.post_distill_weight = -0.1
    with pytest.raises(ValueError, match="post_distill_weight"):
        model_class._validate_geometry_config(None, valid_config)
    valid_config.post_distill_weight = 0.05
    valid_config.text_config.num_hidden_layers = 24
    with pytest.raises(ValueError, match="num_hidden_layers"):
        model_class._validate_geometry_config(None, valid_config)

    image_mask = torch.tensor([[*([True] * 18), False, *([True] * 19)]])
    direct_only_mask = torch.zeros_like(image_mask)
    direct_only_mask[0, :NUM_SPECIAL_TOKENS] = True
    direct_only_mask[0, 19:19 + NUM_SPECIAL_TOKENS] = True
    model_class._validate_expanded_image_spans(
        image_mask,
        direct_only_mask,
        [18, 19],
    )

    direct_only_mask[0, 0] = False
    with pytest.raises(ValueError, match="exactly 17 positions per frame"):
        model_class._validate_expanded_image_spans(
            image_mask,
            direct_only_mask,
            [18, 19],
        )


def test_scene_wrapper_hard_migrates_config_fields(monkeypatch):
    scene_wrapper = _import_qwen35_module_or_skip(
        "qwen_vl.model.modeling_qwen3_5_scene_distill"
    )
    configuration_qwen3_5 = _import_qwen35_module_or_skip(
        "transformers.models.qwen3_5.configuration_qwen3_5"
    )
    monkeypatch.setattr(
        scene_wrapper.Qwen3_5ModelWithGeometry,
        "__init__",
        lambda self, config: None,
    )
    config = configuration_qwen3_5.Qwen3_5Config()
    config.distill_weight = 0.2

    scene_wrapper.Qwen3_5ModelWithSceneDistill(config)

    assert not hasattr(config, "distill_weight")
    assert config.pre_distill_weight == PRE_DISTILL_WEIGHT
    assert config.post_distill_weight == POST_DISTILL_WEIGHT
    saved_config = config.to_dict()
    assert "distill_weight" not in saved_config
    assert saved_config["pre_distill_weight"] == PRE_DISTILL_WEIGHT
    assert saved_config["post_distill_weight"] == POST_DISTILL_WEIGHT


def test_scene_distill_teacher_is_constructed_only_with_a_training_path(monkeypatch):
    scene_wrapper = _import_qwen35_module_or_skip(
        "qwen_vl.model.modeling_qwen3_5_scene_distill"
    )
    model_class = scene_wrapper.Qwen3_5ModelWithSceneDistill
    teacher = nn.Identity()
    teacher_paths = []

    def create_teacher(**kwargs):
        teacher_paths.append(kwargs["model_path"])
        return teacher

    monkeypatch.setattr(scene_wrapper, "create_geometry_encoder", create_teacher)
    monkeypatch.setattr(scene_wrapper, "SceneDistillPreModule", lambda **kwargs: nn.Identity())
    monkeypatch.setattr(scene_wrapper, "SceneDistillPostModule", lambda **kwargs: nn.Identity())

    def make_model(geometry_encoder_path):
        model = object.__new__(model_class)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            geometry_encoder_type="scene_distill",
            geometry_encoder_path=geometry_encoder_path,
            geometry_token_insert_position="front",
            geometry_direct_token_mode="special17",
            reference_frame="first",
            geometry_encoder_freeze=True,
            pre_distill_weight=PRE_DISTILL_WEIGHT,
            post_distill_weight=POST_DISTILL_WEIGHT,
            vision_config=SimpleNamespace(hidden_size=6),
            text_config=SimpleNamespace(hidden_size=8, num_hidden_layers=25),
        )
        model.geometry_encoder = None
        model.language_model = nn.Module()
        model._geometry_modules_initialized = False
        return model

    eval_model = make_model(None)
    eval_model.initialize_geometry_modules()
    assert eval_model.geometry_encoder is None
    assert set(eval_model.language_model.scene_distill) == {"pre", "post"}
    assert teacher_paths == []
    with pytest.raises(RuntimeError, match="training requires a VGGT-Omega teacher"):
        eval_model._collect_teacher_features([torch.zeros(1)], torch.device("cpu"))

    train_model = make_model("facebook/VGGT-Omega")
    train_model.initialize_geometry_modules()
    assert train_model.geometry_encoder is teacher
    assert teacher_paths == ["facebook/VGGT-Omega"]


def test_outer_wrapper_applies_independent_losses_and_clears_transients():
    scene_wrapper = _import_qwen35_module_or_skip(
        "qwen_vl.model.modeling_qwen3_5_scene_distill"
    )
    output_type = scene_wrapper.Qwen3_5ModelOutputWithPast

    class FakeInnerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._last_pre_distill_loss = None
            self._last_post_distill_loss = None
            self.calls = []

        def forward(self, **kwargs):
            compute_pre = kwargs["compute_pre_distill_loss"]
            compute_post = kwargs["compute_post_distill_loss"]
            self.calls.append((compute_pre, compute_post))
            self._last_pre_distill_loss = torch.tensor(2.0) if compute_pre else None
            self._last_post_distill_loss = torch.tensor(5.0) if compute_post else None
            return output_type(last_hidden_state=torch.zeros(1, 2, 4))

    class WrapperHarness(
        scene_wrapper.Qwen3_5ForConditionalGenerationWithSceneDistill
    ):
        @property
        def loss_function(self):
            return lambda **kwargs: torch.tensor(3.0)

    wrapper = object.__new__(WrapperHarness)
    nn.Module.__init__(wrapper)
    wrapper.config = SimpleNamespace(
        geometry_encoder_type="scene_distill",
        image_token_id=99,
        pre_distill_weight=0.0,
        post_distill_weight=0.0,
        text_config=SimpleNamespace(vocab_size=4),
    )
    wrapper.model = FakeInnerModel()
    wrapper.lm_head = nn.Identity()

    labels = torch.zeros(1, 2, dtype=torch.long)
    visual_inputs = {
        "labels": labels,
        "pixel_values": torch.empty(1),
        "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
    }
    for pre_weight, post_weight, expected_loss in (
        (0.0, POST_DISTILL_WEIGHT, 3.0 + POST_DISTILL_WEIGHT * 5.0),
        (PRE_DISTILL_WEIGHT, 0.0, 3.0 + PRE_DISTILL_WEIGHT * 2.0),
        (
            PRE_DISTILL_WEIGHT,
            POST_DISTILL_WEIGHT,
            3.0 + PRE_DISTILL_WEIGHT * 2.0 + POST_DISTILL_WEIGHT * 5.0,
        ),
    ):
        wrapper.config.pre_distill_weight = pre_weight
        wrapper.config.post_distill_weight = post_weight
        outputs = wrapper(**visual_inputs)
        torch.testing.assert_close(outputs.loss, torch.tensor(expected_loss))
        if pre_weight > 0:
            torch.testing.assert_close(
                outputs.pre_distill_cosine_loss, torch.tensor(2.0)
            )
        else:
            assert outputs.pre_distill_cosine_loss is None
        if post_weight > 0:
            torch.testing.assert_close(
                outputs.post_distill_cosine_loss, torch.tensor(5.0)
            )
        else:
            assert outputs.post_distill_cosine_loss is None
        assert wrapper.model.calls[-1] == (pre_weight > 0, post_weight > 0)
        assert wrapper.model._last_pre_distill_loss is None
        assert wrapper.model._last_post_distill_loss is None

    wrapper.eval()
    outputs = wrapper(**visual_inputs)
    torch.testing.assert_close(outputs.loss, torch.tensor(3.0))
    assert outputs.pre_distill_cosine_loss is None
    assert outputs.post_distill_cosine_loss is None
    assert wrapper.model.calls[-1] == (False, False)

    outputs = wrapper(
        labels=None,
        pixel_values=torch.empty(1),
        image_grid_thw=torch.ones(1, 3, dtype=torch.long),
    )
    assert outputs.loss is None
    assert outputs.pre_distill_cosine_loss is None
    assert outputs.post_distill_cosine_loss is None
    assert wrapper.model.calls[-1] == (False, False)


def test_scene_distill_trainer_logs_per_token_cosine_losses(monkeypatch):
    trainer_module = __import__("qwen_vl.train.trainer", fromlist=["*"])
    trainer = object.__new__(trainer_module.SceneDistillTrainer)
    trainer._scene_distill_loss_totals = {}
    trainer._scene_distill_loss_counts = {}
    remote_stats = iter(([10.0, 2.0], [20.0, 4.0]))
    trainer.accelerator = SimpleNamespace(
        gather=lambda value: torch.cat((value, value.new_tensor(next(remote_stats))))
    )

    outputs = SimpleNamespace(
        pre_distill_cosine_loss=torch.tensor(2.0),
        post_distill_cosine_loss=torch.tensor(5.0),
    )
    monkeypatch.setattr(
        trainer_module.Trainer,
        "compute_loss",
        lambda *args, **kwargs: (torch.tensor(3.0), outputs),
    )

    trainer.compute_loss(model=None, inputs={})
    outputs.pre_distill_cosine_loss = torch.tensor(4.0)
    outputs.post_distill_cosine_loss = torch.tensor(7.0)
    trainer.compute_loss(model=None, inputs={})

    logged = {}
    monkeypatch.setattr(
        trainer_module.Trainer,
        "log",
        lambda self, logs, start_time=None: logged.update(logs),
    )
    trainer.log({"loss": 3.0})
    assert logged["loss"] == 3.0
    assert logged["pre_distill_cosine_loss"] == pytest.approx(4.0 / NUM_SPECIAL_TOKENS)
    assert logged["post_distill_cosine_loss"] == pytest.approx(
        32.0 / (6.0 * NUM_SPECIAL_TOKENS)
    )
    assert trainer._pop_scene_distill_logs() == {}


def test_inner_wrapper_always_runs_post_and_reuses_teacher():
    scene_wrapper = _import_qwen35_module_or_skip(
        "qwen_vl.model.modeling_qwen3_5_scene_distill"
    )
    model_class = scene_wrapper.Qwen3_5ModelWithSceneDistill
    output_type = scene_wrapper.Qwen3_5ModelOutputWithPast
    hidden_dim = 8
    teacher_features = torch.randn(1, NUM_SPECIAL_TOKENS, 2048)

    class FakePreModule(nn.Module):
        def forward(self, visual_layer_outputs, frame_sizes, video_sizes):
            return (
                torch.zeros(1, NUM_SPECIAL_TOKENS, hidden_dim),
                teacher_features.clone(),
                torch.zeros(1, NUM_SPECIAL_TOKENS, 1024),
            )

    class FakePostModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_indices = LLM_BLOCK_INDICES

    class FakeLanguageModel(nn.Module):
        def __init__(self, pre_module, post_module):
            super().__init__()
            self.scene_distill = nn.ModuleDict(
                {
                    "pre": pre_module,
                    "post": post_module,
                }
            )
            self.post_requests = []

        def forward(
            self,
            inputs_embeds,
            return_scene_distill_post_features=False,
            **kwargs,
        ):
            self.post_requests.append(self.scene_distill["post"])
            return output_type(
                last_hidden_state=inputs_embeds,
                hidden_states=(teacher_features[..., :1024], teacher_features[..., 1024:])
                if return_scene_distill_post_features
                else None,
            )

    model = object.__new__(model_class)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        image_token_id=99,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    model.geometry_encoder = nn.Identity()
    model.language_model = FakeLanguageModel(FakePreModule(), FakePostModule())
    model._last_pre_distill_loss = None
    model._last_post_distill_loss = None
    model._direct_only_mask = None
    model.rope_deltas = None
    embedding = nn.Embedding(100, hidden_dim)
    model.get_input_embeddings = lambda: embedding
    model._is_scene_distill = lambda: True
    model.align_geometry_modules = lambda reference_tensor=None: None
    vision_hidden_states = [torch.zeros(4, 6) for _ in range(14)]
    model.get_image_features = lambda *args, **kwargs: SimpleNamespace(
        hidden_states=vision_hidden_states,
        pooler_output=torch.zeros(1, hidden_dim),
    )
    model.get_placeholder_mask = lambda input_ids, inputs_embeds, **kwargs: (
        (input_ids == 99).unsqueeze(-1).expand_as(inputs_embeds),
        None,
    )
    teacher_calls = {"count": 0}

    def collect_teacher_features(geometry_encoder_inputs, target_device):
        teacher_calls["count"] += 1
        return teacher_features.to(target_device)

    model._collect_teacher_features = collect_teacher_features
    forward_inputs = {
        "input_ids": torch.tensor([[99]]),
        "attention_mask": torch.ones(1, 1, dtype=torch.long),
        "position_ids": torch.zeros(4, 1, 1, dtype=torch.long),
        "pixel_values": torch.zeros(4, 6),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "geometry_encoder_inputs": [torch.zeros(1, 3, 2, 2)],
    }

    outputs = model(
        **forward_inputs,
        compute_pre_distill_loss=True,
        compute_post_distill_loss=True,
    )
    assert teacher_calls["count"] == 1
    assert model.language_model.post_requests[-1] is model.language_model.scene_distill["post"]
    assert model._last_pre_distill_loss is not None
    assert model._last_post_distill_loss is not None
    assert outputs.hidden_states is None

    teacher_calls["count"] = 0
    model(
        **forward_inputs,
        compute_pre_distill_loss=False,
        compute_post_distill_loss=True,
    )
    assert teacher_calls["count"] == 1
    assert model.language_model.post_requests[-1] is model.language_model.scene_distill["post"]

    teacher_calls["count"] = 0
    model(
        **forward_inputs,
        compute_pre_distill_loss=True,
        compute_post_distill_loss=False,
    )
    assert teacher_calls["count"] == 1
    assert model.language_model.post_requests[-1] is model.language_model.scene_distill["post"]

    teacher_calls["count"] = 0
    model.geometry_encoder = None
    model(
        **forward_inputs,
        compute_pre_distill_loss=False,
        compute_post_distill_loss=False,
    )
    assert teacher_calls["count"] == 0
    assert model.language_model.post_requests[-1] is model.language_model.scene_distill["post"]

    calls_before_decode = len(model.language_model.post_requests)
    model(
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        position_ids=torch.zeros(4, 1, 1, dtype=torch.long),
        cache_position=torch.tensor([1]),
    )
    assert len(model.language_model.post_requests) == calls_before_decode
