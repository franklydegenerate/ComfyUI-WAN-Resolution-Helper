# custom_nodes/wan_resolution_helper.py
# WAN 2.2 Image→Video: Resolution Helper (16x) — Profile-based, adaptive caps
#
# WHAT'S NEW (vs previous):
# - "Profile" dropdown: Highest Quality (slowest), Balanced (regular), Speed (fastest), Custom
# - Adaptive cap selection: picks a cap from a profile "ladder" CLOSEST to the image's long side,
#   respecting per-profile upscale limits and a global 1280 cap (Custom can exceed 1280).
# - Extreme aspect ratio handling: safeguards against overly tiny short sides; will switch to a
#   short-side-first strategy if needed.
# - Tooltips/hover help for every control (supported in recent ComfyUI; if your build ignores
#   tooltips, it won’t break anything).
#
# OUTPUTS:
# - width_out (INT), height_out (INT): pass straight to a Resize node
# - info (STRING): readable summary of what was chosen and why
# - profile_note (STRING): short explainer of the selected profile

from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import math

CATEGORY = "WAN/Resolution"

# -----------------------------
# Profiles & UI Help
# -----------------------------

PROFILES = [
    # (key, display name)
    ("highest_quality", "Highest Quality (slowest)"),
    ("balanced", "Balanced (regular)"),
    ("speed", "Speed (fastest)"),
    ("custom", "Custom (manual)"),
]

PROFILE_HELP: Dict[str, str] = {
    "highest_quality": (
        "Targets up to 1280 on the long side (divisible by 16). Adaptive: picks the closest sensible cap "
        "to your image's long side without exceeding a ~2.0× upscale. Enforces a minimum short side ≈512 "
        "via fallback if the aspect ratio is extreme."
    ),
    "balanced": (
        "Targets ~1024/960/896 on the long side. Adaptive: chooses the closest sensible cap without exceeding "
        "a ~1.5× upscale. Minimum short side ≈448 with fallback for extreme ratios."
    ),
    "speed": (
        "Targets ~896/832/768/704/640 on the long side. Adaptive: chooses the closest sensible cap without "
        "exceeding ~1.25× upscale. Minimum short side ≈384 with fallback for extreme ratios—best for previews."
    ),
    "custom": (
        "You set the max side. Can exceed 1280 if desired. Still snaps both sides to multiples of 16 and preserves "
        "aspect ratio. Extreme-AR fallback applies if enabled."
    ),
}

# Per-profile configuration:
# - ladder: candidate long-side caps (will add the snapped original long side as another candidate)
# - max_upscale: do not upscale beyond this factor (relative to original long side)
# - min_short: target minimum short side (after scaling) to avoid brittle thin sides
# - global_cap: maximum allowed cap (None = unlimited)
PROFILE_CFG = {
    "highest_quality": {
        "ladder": [1280, 1152, 1088, 1024, 960],
        "max_upscale": 2.0,
        "min_short": 512,
        "global_cap": 1280,
    },
    "balanced": {
        "ladder": [1024, 960, 896],
        "max_upscale": 1.5,
        "min_short": 448,
        "global_cap": 1280,
    },
    "speed": {
        "ladder": [896, 832, 768, 704, 640],
        "max_upscale": 1.25,
        "min_short": 384,
        "global_cap": 1280,
    },
    "custom": {
        "ladder": [],  # filled at runtime with [custom_max_side]
        "max_upscale": None,  # no extra limits beyond scale_behavior
        "min_short": 384,     # still keep a gentle floor for extreme AR unless disabled
        "global_cap": None,   # no hard cap unless user’s custom value acts as one
    },
}

ROUNDING_HELP = (
    "Rounding when snapping to multiples of 16:\n"
    "• nearest: pick the closest multiple of 16\n"
    "• floor: always round down to the next lower multiple of 16\n"
    "• ceil: always round up to the next higher multiple of 16"
)

SCALE_BEHAVIOR_CHOICES = [
    ("scale_to_target", "Scale to target (up/down)"),
    ("downscale_only", "Downscale only"),
]

SCALE_BEHAVIOR_HELP = (
    "Scale to target: the long side may be up- or down-scaled (subject to the profile’s upscale limit).\n"
    "Downscale only: never upscale; the target cap acts as a ceiling only."
)

EXTREME_AR_HELP = (
    "When the image is very tall or wide, the computed short side could get too small. "
    "With this ON, the node will switch to a short-side-first strategy (with sensible floors) "
    "to avoid brittle outputs."
)

CUSTOM_MAX_SIDE_HELP = (
    "Only used when Profile = Custom. Sets the desired long-side cap (divisible by 16). "
    "Can exceed 1280 if your pipeline/VRAM allows."
)

# -----------------------------
# Utility functions
# -----------------------------

def _label_orientation(w: int, h: int) -> str:
    if w == h:
        return "square"
    return "landscape" if w > h else "portrait"

def _snap16(n: float, mode: str = "nearest", cap: Optional[int] = None) -> int:
    if n <= 0:
        return 16
    if mode == "floor":
        out = int(n // 16) * 16
    elif mode == "ceil":
        out = int(math.ceil(n / 16.0)) * 16
    else:
        out = int(round(n / 16.0)) * 16
    if out < 16:
        out = 16
    if cap is not None and out > cap:
        out = cap if cap % 16 == 0 else (cap // 16) * 16
    return out

def _build_candidates(
    profile_key: str,
    L: int,
    custom_max_side: Optional[int],
    include_snapped_original: bool,
) -> List[int]:
    cfg = PROFILE_CFG[profile_key]
    ladder = list(cfg["ladder"])
    if profile_key == "custom":
        if custom_max_side is None or custom_max_side < 16:
            ladder = [1280]  # safe default if user forgot to set
        else:
            ladder = [int(custom_max_side)]
    # Always consider the snapped original long side (it respects the native scale)
    if include_snapped_original:
        ladder.append(_snap16(L, mode="nearest"))
    # Deduplicate and sort descending (we want to gravitate to larger caps first)
    uniq = sorted({int(x) for x in ladder if x >= 16}, reverse=True)
    return uniq

def _pick_cap_for_profile(
    profile_key: str,
    L: int,
    S: int,
    rounding_mode: str,
    scale_behavior: str,
    custom_max_side: Optional[int],
) -> Tuple[int, Dict[str, Any]]:
    """
    Returns (chosen_cap, debug_info)
    """
    cfg = PROFILE_CFG[profile_key]
    global_cap = cfg["global_cap"]
    max_upscale = cfg["max_upscale"]
    min_short = cfg["min_short"]

    candidates = _build_candidates(profile_key, L, custom_max_side, include_snapped_original=True)

    # Apply global cap (except Custom where global_cap=None)
    def clamp_global(cap: int) -> int:
        if global_cap is None:
            return cap
        return min(cap, global_cap)

    # Filter candidates by scale behavior and upscale limit
    filtered: List[int] = []
    for c in candidates:
        c = clamp_global(c)
        if scale_behavior == "downscale_only" and c > L:
            continue
        if max_upscale is not None and c > L * max_upscale:
            continue
        filtered.append(c)

    if not filtered:
        # If everything filtered out, fall back to the safest option:
        # - if downscale_only: use min(L, global_cap) snapped
        # - else: allow the smallest candidate even if it violates upscale limit (as last resort)
        if scale_behavior == "downscale_only":
            fallback = _snap16(min(L, global_cap if global_cap else L), mode=rounding_mode)
            return fallback, {
                "reason": "fallback_downscale_only",
                "candidates": candidates,
                "filtered": filtered,
            }
        else:
            # permit the closest candidate regardless of upscale limit (rare)
            closest = min(candidates, key=lambda c: abs(c - L))
            closest = clamp_global(closest)
            return closest, {
                "reason": "fallback_closest",
                "candidates": candidates,
                "filtered": filtered,
            }

    # Choose the candidate closest to L
    chosen = min(filtered, key=lambda c: abs(c - L))
    return chosen, {
        "reason": "closest_to_input_long_side",
        "candidates": candidates,
        "filtered": filtered,
    }

def _short_side_fallback_targets(profile_key: str) -> List[int]:
    """Short-side-first floors to consider when AR is extreme."""
    cfg = PROFILE_CFG[profile_key]
    base = max(16, int(cfg["min_short"]))
    # Build a small, sensible ladder around that floor (descending)
    ladder = sorted({base, base + 64, base + 128, base + 192, base + 256}, reverse=True)
    # Make them multiples of 16
    ladder = [_snap16(x, mode="nearest") for x in ladder]
    ladder = sorted(set(ladder), reverse=True)
    return ladder

def _compute_scaled_dims_from_long(
    w: int, h: int, long_target: int, rounding_mode: str, cap: Optional[int]
) -> Tuple[int, int]:
    L_in, S_in = (w, h) if w >= h else (h, w)
    aspect = L_in / float(S_in)  # >= 1.0
    long_snapped = _snap16(long_target, mode=rounding_mode, cap=cap)
    short_float = long_snapped / aspect
    short_snapped = _snap16(short_float, mode=rounding_mode)
    if w >= h:
        return int(long_snapped), int(short_snapped)
    else:
        return int(short_snapped), int(long_snapped)

def _compute_scaled_dims_short_first(
    w: int, h: int, short_target: int, rounding_mode: str, cap_long: Optional[int]
) -> Tuple[int, int]:
    L_in, S_in = (w, h) if w >= h else (h, w)
    aspect = L_in / float(S_in)  # >= 1.0
    short_snapped = _snap16(short_target, mode=rounding_mode)
    long_float = short_snapped * aspect
    long_snapped = _snap16(long_float, mode=rounding_mode, cap=cap_long)
    if w >= h:
        return int(long_snapped), int(short_snapped)
    else:
        return int(short_snapped), int(long_snapped)

def _compute_optimal(
    w: int,
    h: int,
    profile_key: str,
    rounding_mode: str,
    scale_behavior: str,
    use_extreme_ar_handling: bool,
    custom_max_side: Optional[int],
) -> Tuple[int, int, Dict[str, Any]]:
    if w <= 0 or h <= 0:
        raise ValueError("Input image dimensions must be positive.")

    ori_in = _label_orientation(w, h)
    L_in, S_in = (w, h) if w >= h else (h, w)
    cfg = PROFILE_CFG[profile_key]
    global_cap = cfg["global_cap"]
    min_short = cfg["min_short"]

    chosen_cap, pick_info = _pick_cap_for_profile(
        profile_key, L_in, S_in, rounding_mode, scale_behavior, custom_max_side
    )

    cap16 = (chosen_cap // 16) * 16 if chosen_cap is not None else None
    if global_cap is not None and cap16 is not None:
        cap16 = min(cap16, (global_cap // 16) * 16)

    # Long-side-first attempt
    out_w, out_h = _compute_scaled_dims_from_long(w, h, chosen_cap, rounding_mode, cap16)
    S_out = min(out_w, out_h)

    used_short_side_fallback = False

    # Extreme-AR fallback if the short side is too small
    if use_extreme_ar_handling and min_short and S_out < min_short:
        for short_target in _short_side_fallback_targets(profile_key):
            out_w2, out_h2 = _compute_scaled_dims_short_first(w, h, short_target, rounding_mode, cap16)
            if min(out_w2, out_h2) >= min_short:
                out_w, out_h = out_w2, out_h2
                used_short_side_fallback = True
                break
        # if still not met, keep the best we got (out_w, out_h)

    info = {
        "input_w": w,
        "input_h": h,
        "orientation_in": ori_in,
        "profile": profile_key,
        "chosen_cap": chosen_cap,
        "cap16": cap16,
        "pick_info": pick_info,
        "min_short": min_short,
        "used_short_side_fallback": used_short_side_fallback,
        "out_w": out_w,
        "out_h": out_h,
        "orientation_out": _label_orientation(out_w, out_h),
    }
    return out_w, out_h, info

# -----------------------------
# ComfyUI Node
# -----------------------------

class WANResolutionHelperNodeV2:
    """
    ComfyUI node:
      Inputs:
        - image (IMAGE)
        - profile (CHOICE): Highest Quality / Balanced / Speed / Custom
        - custom_max_side (INT): only used for Custom
        - rounding_mode (CHOICE)
        - scale_behavior (CHOICE): Scale to target / Downscale only
        - extreme_ar_handling (BOOLEAN)
      Outputs:
        - width_out (INT), height_out (INT)
        - info (STRING)        : detailed summary
        - profile_note (STRING): short explainer for the chosen profile
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Build profile choices for UI
        profile_labels = [label for key, label in PROFILES]
        profile_keys = [key for key, _ in PROFILES]

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image tensor [B,H,W,C]. Only the first in the batch is read."}),
                "profile": (
                    profile_labels,
                    {
                        "default": "Highest Quality (slowest)",
                        "tooltip": (
                            "Choose a profile:\n"
                            "• Highest Quality (slowest): aims near 1280, adaptive to your image, ≤~2.0× upscale\n"
                            "• Balanced (regular): aims near 1024/960/896, ≤~1.5× upscale\n"
                            "• Speed (fastest): aims near 896/832/768..., ≤~1.25× upscale\n"
                            "• Custom: you set the cap (can exceed 1280)"
                        ),
                    },
                ),
                "custom_max_side": (
                    "INT",
                    {
                        "default": 1280,
                        "min": 16,
                        "max": 8192,
                        "step": 16,
                        "tooltip": CUSTOM_MAX_SIDE_HELP,
                    },
                ),
                "rounding_mode": (
                    ["nearest", "floor", "ceil"],
                    {"default": "nearest", "tooltip": ROUNDING_HELP},
                ),
                "scale_behavior": (
                    [label for _, label in SCALE_BEHAVIOR_CHOICES],
                    {"default": "Scale to target (up/down)", "tooltip": SCALE_BEHAVIOR_HELP},
                ),
                "extreme_ar_handling": (
                    "BOOLEAN",
                    {"default": True, "tooltip": EXTREME_AR_HELP},
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("width_out", "height_out", "info", "profile_note")
    FUNCTION = "compute"
    CATEGORY = CATEGORY

    # Helper: map label -> key
    _PROFILE_LABEL_TO_KEY = {label: key for key, label in PROFILES}
    _SCALE_LABEL_TO_KEY = {label: key for key, label in SCALE_BEHAVIOR_CHOICES}

    def compute(
        self,
        image,
        profile,
        custom_max_side,
        rounding_mode,
        scale_behavior,
        extreme_ar_handling,
    ):
        import torch

        # Map UI labels back to keys
        profile_key = self._PROFILE_LABEL_TO_KEY.get(profile, "balanced")
        scale_key = self._SCALE_LABEL_TO_KEY.get(scale_behavior, "scale_to_target")

        # Pull width/height from image
        if image is None:
            raise ValueError("No image provided.")
        if isinstance(image, torch.Tensor):
            if image.ndim != 4 or image.shape[-1] not in (1, 3, 4):
                raise ValueError(
                    f"Unexpected image tensor shape {tuple(image.shape)}. "
                    "Expected [batch, height, width, channels]."
                )
            _, h, w, _ = image.shape
        else:
            try:
                w, h = image.size
            except Exception as e:
                raise ValueError("Unsupported image type for this node.") from e

        out_w, out_h, info = _compute_optimal(
            int(w),
            int(h),
            profile_key=profile_key,
            rounding_mode=str(rounding_mode),
            scale_behavior=str(scale_key),
            use_extreme_ar_handling=bool(extreme_ar_handling),
            custom_max_side=int(custom_max_side) if profile_key == "custom" else None,
        )

        # Build human-readable strings
        profile_note = PROFILE_HELP.get(profile_key, "")
        chosen = info.get("chosen_cap")
        cap16 = info.get("cap16")
        reason = info.get("pick_info", {}).get("reason")

        summary_lines = [
            f"Profile: {profile}  |  Rounding: {rounding_mode}  |  {scale_behavior}",
            f"Input: {w}×{h} ({_label_orientation(w,h)})",
        ]
        if profile_key == "custom":
            summary_lines.append(f"Custom cap requested: {custom_max_side}  |  cap16={cap16}")
        else:
            summary_lines.append(f"Chosen cap: {chosen}  (cap16={cap16}, reason={reason})")

        summary_lines.append(
            f"Output (16x): {out_w}×{out_h} → {_label_orientation(out_w,out_h)}"
        )
        if info.get("used_short_side_fallback"):
            summary_lines.append("Note: Extreme-AR fallback (short-side-first) was used to protect the short side.")

        info_str = "\n".join(summary_lines)

        return (int(out_w), int(out_h), info_str, profile_note)


NODE_CLASS_MAPPINGS = {
    "WANResolutionHelperV2": WANResolutionHelperNodeV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WANResolutionHelperV2": "WAN 2.2 Resolution Helper (16x, Profiles)",
}