"""Shared config helpers for VGGT-Omega direct-injection geometry paths."""

from __future__ import annotations


VGGT_OMEGA_DIRECT_ENCODER_TYPES = {"vggt_omega_direct"}
VGGT_OMEGA_DIRECT_TOKEN_MODE_CAMERA = "camera"
VGGT_OMEGA_DIRECT_TOKEN_MODE_SCENE16 = "scene16"
VGGT_OMEGA_DIRECT_TOKEN_MODE_SPECIAL17 = "special17"

_DIRECT_TOKEN_MODE_TO_COUNT = {
    VGGT_OMEGA_DIRECT_TOKEN_MODE_CAMERA: 1,
    VGGT_OMEGA_DIRECT_TOKEN_MODE_SCENE16: 16,
    VGGT_OMEGA_DIRECT_TOKEN_MODE_SPECIAL17: 17,
}


def is_vggt_omega_direct_geometry_encoder(geometry_encoder_type: str) -> bool:
    return str(geometry_encoder_type or "").strip().lower() in VGGT_OMEGA_DIRECT_ENCODER_TYPES


def resolve_vggt_omega_direct_token_mode(
    geometry_encoder_type: str,
    direct_token_mode: str | None = None,
) -> str:
    encoder_type = str(geometry_encoder_type or "").strip().lower()
    requested_mode = str(direct_token_mode or "").strip().lower()

    if encoder_type != "vggt_omega_direct":
        if requested_mode:
            raise ValueError(
                "`geometry_direct_token_mode` is only valid when `geometry_encoder_type` is "
                f"`vggt_omega_direct`. Got encoder_type={geometry_encoder_type!r}."
            )
        return VGGT_OMEGA_DIRECT_TOKEN_MODE_SPECIAL17

    if not requested_mode:
        return VGGT_OMEGA_DIRECT_TOKEN_MODE_SPECIAL17

    if requested_mode not in _DIRECT_TOKEN_MODE_TO_COUNT:
        raise ValueError(
            f"Unsupported geometry_direct_token_mode={direct_token_mode!r}; expected one of "
            f"{sorted(_DIRECT_TOKEN_MODE_TO_COUNT)}."
        )
    return requested_mode


def get_vggt_omega_direct_num_extra_tokens(
    geometry_encoder_type: str,
    direct_token_mode: str | None = None,
) -> int:
    if not is_vggt_omega_direct_geometry_encoder(geometry_encoder_type):
        return 0
    token_mode = resolve_vggt_omega_direct_token_mode(geometry_encoder_type, direct_token_mode)
    return _DIRECT_TOKEN_MODE_TO_COUNT[token_mode]


__all__ = [
    "VGGT_OMEGA_DIRECT_ENCODER_TYPES",
    "VGGT_OMEGA_DIRECT_TOKEN_MODE_CAMERA",
    "VGGT_OMEGA_DIRECT_TOKEN_MODE_SCENE16",
    "VGGT_OMEGA_DIRECT_TOKEN_MODE_SPECIAL17",
    "get_vggt_omega_direct_num_extra_tokens",
    "is_vggt_omega_direct_geometry_encoder",
    "resolve_vggt_omega_direct_token_mode",
]
