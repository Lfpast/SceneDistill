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
    remove_teacher_weights,
    scene_distillation_loss,
    select_pre_vision_layer_outputs,
)
from qwen_vl.model.vggt_omega_direct_packing import (
    expand_image_embeds_with_direct_tokens,
    expand_visual_placeholders,
)


def test_special_token_initialization_uses_first_and_other_variants():
    module = SceneDistillPreModule(visual_dim=6, text_hidden_dim=10, stream_dim=8, num_heads=2)
    with torch.no_grad():
        module.pre_camera_token[:, 0].fill_(1.0)
        module.pre_camera_token[:, 1].fill_(2.0)
        module.pre_scene_token[:, 0].fill_(3.0)
        module.pre_scene_token[:, 1].fill_(4.0)

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

    assert module.pre_camera_token.grad is not None
    assert module.pre_scene_token.grad is not None
    assert module.pre_projector.linear_fc2.weight.grad is not None
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


def test_checkpoint_filter_removes_teacher_but_keeps_student():
    state_dict = {
        "model.geometry_encoder.vggt_omega.weight": torch.ones(1),
        "geometry_encoder.vggt_omega.weight": torch.ones(1),
        "model.scene_distill_pre_module.pre_camera_token": torch.ones(1),
        "model.scene_distill_post_module.post_frame_layers.0.q_proj.weight": torch.ones(1),
        "model.language_model.weight": torch.ones(1),
    }

    filtered = remove_teacher_weights(state_dict)

    assert set(filtered) == {
        "model.scene_distill_pre_module.pre_camera_token",
        "model.scene_distill_post_module.post_frame_layers.0.q_proj.weight",
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

    post_features = module(
        pre_global_tokens,
        llm_layer_features,
        frame_sizes=frame_sizes,
        video_sizes=[2, 1],
    )

    assert LLM_BLOCK_INDICES == (4, 8, 12, 16, 20, 24)
    assert POST_DISTILL_DEPTH == 6
    assert post_features.shape == (3, NUM_SPECIAL_TOKENS, 16)
    assert post_features[..., :8].shape == (3, NUM_SPECIAL_TOKENS, 8)
    assert post_features[..., 8:].shape == (3, NUM_SPECIAL_TOKENS, 8)

    post_features.square().mean().backward()
    assert pre_global_tokens.grad is not None
    assert all(features.grad is not None for features in llm_layer_features)
    assert module.post_frame_layers[0].q_proj.weight.grad is not None
    assert module.post_global_layers[-1].qkv.weight.grad is not None


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
        pre_global_tokens,
        llm_layer_features,
        frame_sizes=frame_sizes,
        video_sizes=[1, 1],
    )
    changed_features = [features.clone() for features in llm_layer_features]
    for features in changed_features:
        features[frame_sizes[0]:] *= -2.0
    changed = module(
        pre_global_tokens,
        changed_features,
        frame_sizes=frame_sizes,
        video_sizes=[1, 1],
    )

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
    assert len({id(layer) for layer in post_module.post_frame_layers}) == POST_DISTILL_DEPTH
    assert len({id(layer) for layer in post_module.post_global_layers}) == POST_DISTILL_DEPTH
    assert all(
        isinstance(layer, FrameCrossAttentionLayer)
        for layer in post_module.post_frame_layers
    )
    assert all(isinstance(layer, GlobalSelfAttentionLayer) for layer in pre_module.pre_global_layers)
    assert all(isinstance(layer, GlobalSelfAttentionLayer) for layer in post_module.post_global_layers)
    assert not any(
        component in name
        for name, _ in post_module.named_parameters()
        for component in ("camera_token", "scene_token", "projector")
    )
    assert all(
        key.startswith(
            (
                "pre_camera_token",
                "pre_scene_token",
                "pre_frame_layers",
                "pre_global_layers",
                "pre_projector",
            )
        )
        for key in pre_module.state_dict()
    )


def test_post_module_rejects_missing_layers_and_invalid_shapes():
    module = SceneDistillPostModule(
        llm_hidden_dim=6,
        special_dim=8,
        num_heads=2,
    )
    pre_global_tokens = torch.randn(2, NUM_SPECIAL_TOKENS, 8)
    frame_sizes = [18, 19]
    valid_features = [
        torch.randn(sum(frame_sizes), 6)
        for _ in LLM_BLOCK_INDICES
    ]

    with pytest.raises(ValueError, match="requires 6 LLM layers"):
        module(
            pre_global_tokens,
            valid_features[:-1],
            frame_sizes=frame_sizes,
            video_sizes=[2],
        )
    with pytest.raises(ValueError, match="features must have shape"):
        module(
            pre_global_tokens,
            [*valid_features[:-1], torch.randn(sum(frame_sizes), 7)],
            frame_sizes=frame_sizes,
            video_sizes=[2],
        )
    with pytest.raises(ValueError, match="pre_global_tokens must have shape"):
        module(
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

    target.load_state_dict(source.state_dict(), strict=True)

    for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
        torch.testing.assert_close(source_parameter, target_parameter)


def _import_qwen35_module_or_skip(module_name):
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as error:
        pytest.skip(f"Qwen3.5 Transformers runtime is unavailable: {error}")


def test_selective_llm_capture_is_masked_layer_input_and_pre_norm(monkeypatch):
    modeling_qwen3_5 = _import_qwen35_module_or_skip(
        "qwen_vl.model.modeling_qwen3_5"
    )
    assert "scene_distill_pre_module" in modeling_qwen3_5.GEOMETRY_STATE_KEYWORDS
    assert "scene_distill_post_module" in modeling_qwen3_5.GEOMETRY_STATE_KEYWORDS
    assert "scene_distill_module" not in modeling_qwen3_5.GEOMETRY_STATE_KEYWORDS

    class DummyDecoderLayer(nn.Module):
        layer_type = "full_attention"

        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, hidden_states, **kwargs):
            return hidden_states + self.value

    class DummyRotaryEmbedding(nn.Module):
        def forward(self, hidden_states, position_ids):
            return None

    class ScaleNorm(nn.Module):
        def forward(self, hidden_states):
            return hidden_states * 10

    text_model = object.__new__(modeling_qwen3_5.Qwen3_5TextModelWithGeometry)
    nn.Module.__init__(text_model)
    text_model.config = SimpleNamespace(num_hidden_layers=25)
    text_model.layers = nn.ModuleList(
        DummyDecoderLayer(layer_index + 1)
        for layer_index in range(text_model.config.num_hidden_layers)
    )
    text_model.rotary_emb = DummyRotaryEmbedding()
    text_model.norm = ScaleNorm()
    text_model._update_linear_attn_mask = lambda attention_mask, cache_position: None
    monkeypatch.setattr(modeling_qwen3_5, "create_causal_mask", lambda **kwargs: None)

    inputs_embeds = torch.zeros(1, 4, 8, requires_grad=True)
    capture_mask = torch.tensor([[True, True, False, True]])
    outputs = text_model(
        inputs_embeds=inputs_embeds,
        use_cache=False,
        cache_position=torch.arange(4),
        capture_hidden_state_layers=LLM_BLOCK_INDICES,
        capture_hidden_state_mask=capture_mask,
    )

    assert len(outputs.hidden_states) == POST_DISTILL_DEPTH
    assert all(features.shape == (3, 8) for features in outputs.hidden_states)
    expected_layer_values = [10, 36, 78, 136, 210, 300]
    for features, expected_value in zip(outputs.hidden_states, expected_layer_values):
        torch.testing.assert_close(features, torch.full_like(features, expected_value))
    torch.testing.assert_close(
        outputs.last_hidden_state,
        torch.full_like(outputs.last_hidden_state, 3250),
    )

    sum(features.sum() for features in outputs.hidden_states).backward()
    assert inputs_embeds.grad[0, 2].abs().sum() == 0
    assert inputs_embeds.grad[0, [0, 1, 3]].abs().sum() > 0

    with pytest.raises(ValueError, match="ascending order"):
        text_model(
            inputs_embeds=torch.zeros(1, 4, 8),
            use_cache=False,
            cache_position=torch.arange(4),
            capture_hidden_state_layers=(8, 4),
            capture_hidden_state_mask=capture_mask,
        )


def test_selective_llm_capture_survives_gradient_checkpointing():
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
    input_ids = torch.randint(0, config.vocab_size, (1, 12))
    capture_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    capture_mask[:, 2:10] = True

    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        capture_hidden_state_layers=LLM_BLOCK_INDICES,
        capture_hidden_state_mask=capture_mask,
    )
    loss = sum(features.square().mean() for features in outputs.hidden_states)
    loss.backward()

    assert model.is_gradient_checkpointing
    assert all(features.shape == (8, config.hidden_size) for features in outputs.hidden_states)
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
        model.scene_distill_pre_module = None
        model.scene_distill_post_module = None
        model._geometry_modules_initialized = False
        return model

    eval_model = make_model(None)
    eval_model.initialize_geometry_modules()
    assert eval_model.geometry_encoder is None
    assert eval_model.scene_distill_pre_module is not None
    assert eval_model.scene_distill_post_module is not None
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
        assert wrapper.model.calls[-1] == (pre_weight > 0, post_weight > 0)
        assert wrapper.model._last_pre_distill_loss is None
        assert wrapper.model._last_post_distill_loss is None

    wrapper.eval()
    outputs = wrapper(**visual_inputs)
    torch.testing.assert_close(outputs.loss, torch.tensor(3.0))
    assert wrapper.model.calls[-1] == (False, False)

    outputs = wrapper(
        labels=None,
        pixel_values=torch.empty(1),
        image_grid_thw=torch.ones(1, 3, dtype=torch.long),
    )
    assert outputs.loss is None
    assert wrapper.model.calls[-1] == (False, False)


def test_inner_wrapper_reuses_teacher_and_only_runs_post_when_enabled():
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
            self.calls = 0

        def forward(self, pre_global_tokens, llm_layer_features, frame_sizes, video_sizes):
            self.calls += 1
            return teacher_features.clone()

    class FakeLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.capture_requests = []

        def forward(
            self,
            inputs_embeds,
            capture_hidden_state_layers=None,
            capture_hidden_state_mask=None,
            **kwargs,
        ):
            self.capture_requests.append(capture_hidden_state_layers)
            captured = None
            if capture_hidden_state_layers is not None:
                captured = tuple(
                    torch.zeros(int(capture_hidden_state_mask.sum().item()), hidden_dim)
                    for _ in capture_hidden_state_layers
                )
            return output_type(
                last_hidden_state=inputs_embeds,
                hidden_states=captured,
            )

    model = object.__new__(model_class)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        image_token_id=99,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    model.geometry_encoder = nn.Identity()
    model.scene_distill_pre_module = FakePreModule()
    model.scene_distill_post_module = FakePostModule()
    model.language_model = FakeLanguageModel()
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
    assert model.scene_distill_post_module.calls == 1
    assert model.language_model.capture_requests[-1] == LLM_BLOCK_INDICES
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
    assert model.scene_distill_post_module.calls == 2

    teacher_calls["count"] = 0
    model(
        **forward_inputs,
        compute_pre_distill_loss=True,
        compute_post_distill_loss=False,
    )
    assert teacher_calls["count"] == 1
    assert model.scene_distill_post_module.calls == 2
    assert model.language_model.capture_requests[-1] is None

    teacher_calls["count"] = 0
    model.geometry_encoder = None
    model(
        **forward_inputs,
        compute_pre_distill_loss=False,
        compute_post_distill_loss=False,
    )
    assert teacher_calls["count"] == 0
    assert model.scene_distill_post_module.calls == 2
