import torch

from qwen_vl.model.scene_distill_module import (
    DISTILL_WEIGHT,
    NUM_SPECIAL_TOKENS,
    VISION_BLOCK_INDICES,
    FrameCrossAttentionLayer,
    GlobalCameraSceneSelfAttentionLayer,
    SceneDistillModule,
    remove_teacher_weights,
    scene_distillation_loss,
    select_vision_layer_outputs,
)
from qwen_vl.model.vggt_omega_direct_packing import (
    expand_image_embeds_with_direct_tokens,
    expand_visual_placeholders,
)


def test_special_token_initialization_uses_first_and_other_variants():
    module = SceneDistillModule(visual_dim=6, text_hidden_dim=10, stream_dim=8, num_heads=2)
    with torch.no_grad():
        module.camera_token[:, 0].fill_(1.0)
        module.camera_token[:, 1].fill_(2.0)
        module.scene_token[:, 0].fill_(3.0)
        module.scene_token[:, 1].fill_(4.0)

    tokens = module.prepare_special_tokens([2, 3])

    assert tokens.shape == (5, NUM_SPECIAL_TOKENS, 8)
    torch.testing.assert_close(tokens[[0, 2], 0], torch.ones(2, 8))
    torch.testing.assert_close(tokens[[1, 3, 4], 0], torch.full((3, 8), 2.0))
    torch.testing.assert_close(tokens[[0, 2], 1:], torch.full((2, 16, 8), 3.0))
    torch.testing.assert_close(tokens[[1, 3, 4], 1:], torch.full((3, 16, 8), 4.0))


def test_frame_cross_attention_isolates_frames():
    torch.manual_seed(0)
    layer = FrameCrossAttentionLayer(special_dim=8, visual_dim=6, num_heads=2).eval()
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
    layer = GlobalCameraSceneSelfAttentionLayer(special_dim=8, num_heads=2).eval()
    special_tokens = torch.randn(3, NUM_SPECIAL_TOKENS, 8)

    baseline = layer(special_tokens, video_sizes=[2, 1])
    changed_tokens = special_tokens.clone()
    changed_tokens[2] += 10.0
    changed = layer(changed_tokens, video_sizes=[2, 1])

    torch.testing.assert_close(baseline[:2], changed[:2])
    assert not torch.allclose(baseline[2], changed[2])


def test_four_stage_module_shapes():
    torch.manual_seed(2)
    module = SceneDistillModule(visual_dim=6, text_hidden_dim=10, stream_dim=8, num_heads=2)
    visual_layers = [torch.randn(7, 6) for _ in VISION_BLOCK_INDICES]

    embeds, features = module(visual_layers, frame_sizes=[2, 3, 2], video_sizes=[2, 1])

    assert VISION_BLOCK_INDICES == (0, 4, 8, 12)
    assert features.shape == (3, NUM_SPECIAL_TOKENS, 16)
    assert embeds.shape == (3, NUM_SPECIAL_TOKENS, 10)


def test_gcte_detaches_vision_and_trains_student_parameters():
    torch.manual_seed(4)
    module = SceneDistillModule(visual_dim=6, text_hidden_dim=10, stream_dim=8, num_heads=2)
    visual_layers = [torch.randn(4, 6, requires_grad=True) for _ in VISION_BLOCK_INDICES]

    embeds, features = module(visual_layers, frame_sizes=[2, 2], video_sizes=[2])
    (embeds.sum() + features.sum()).backward()

    assert module.camera_token.grad is not None
    assert module.scene_token.grad is not None
    assert module.projector.linear_fc2.weight.grad is not None
    assert all(visual_features.grad is None for visual_features in visual_layers)


def test_selects_exact_vision_block_outputs():
    hidden_states = [torch.full((1, 2), float(index)) for index in range(24)]

    selected = select_vision_layer_outputs(hidden_states)

    assert [int(features[0, 0].item()) for features in selected] == [0, 4, 8, 12]


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
    total_loss = sft_loss + DISTILL_WEIGHT * distill_loss
    total_loss.backward()

    assert student.grad is not None
    assert teacher.grad is None
    torch.testing.assert_close(total_loss.detach(), 3.0 + DISTILL_WEIGHT * distill_loss.detach())


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


def test_scene_distill_module_state_dict_round_trip():
    torch.manual_seed(3)
    source = SceneDistillModule(visual_dim=6, text_hidden_dim=10, stream_dim=8, num_heads=2)
    target = SceneDistillModule(visual_dim=6, text_hidden_dim=10, stream_dim=8, num_heads=2)

    target.load_state_dict(source.state_dict(), strict=True)

    for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
        torch.testing.assert_close(source_parameter, target_parameter)


def test_checkpoint_filter_removes_teacher_but_keeps_student():
    state_dict = {
        "model.geometry_encoder.vggt_omega.weight": torch.ones(1),
        "geometry_encoder.vggt_omega.weight": torch.ones(1),
        "model.scene_distill_module.camera_token": torch.ones(1),
        "model.language_model.weight": torch.ones(1),
    }

    filtered = remove_teacher_weights(state_dict)

    assert set(filtered) == {
        "model.scene_distill_module.camera_token",
        "model.language_model.weight",
    }
