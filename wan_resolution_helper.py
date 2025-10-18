# custom_nodes/wan_resolution_helper.py
# WAN 2.2 Image→Video: Resolution Helper (16x) — Profiles, Downscale-only, AR output

from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import math

CATEGORY = "WAN/Resolution"

# -----------------------------
# Profiles & UI Help
# -----------------------------

PROFILES = [
    ("highest_quality", "Highest Quality (slowest)"),
    ("balanced", "Balanced (regular)"),
    ("speed", "Speed (fastest)"),
    ("custom", "Custom (manual)"),
]

PROFILE_HELP: Dict[str, str] = {
    "highest_quality": (
        "Targets up to 1280 on the long side (divisible by 16). Adaptive: picks the closest sensible cap "
        "to your image's long side. Enforces a minimum short side ≈512 via fallback if the aspect ratio is extreme. "
        "Strictly downscale-only (never upsizes)."
    ),
    "balanced": (
        "Targets ~1024/960/896 on the long side (never above 1024). Adaptive: chooses the closest sensible cap. "
        "Minimum short side ≈448 with fallback for extreme ratios. Strictly downscale-only."
    ),
    "speed": (
        "Targets ~896/832/768/704/640 on the long side (never above 896). Adaptive: chooses the closest sensible cap. "
        "Minimum short side ≈384 with fallback for extreme ratios—best for previews. Strictly downscale-only."
    ),
    "custom": (
        "You set the long-side cap. Can exceed 1280 if desired. Preserves aspect ratio and snaps both sides to multiples "
        "of 16. Always downscales toward your target (won’t upscale). Extreme-AR fallback applies if enabled."
    ),
}

# Per-profile configuration:
# - ladder: candidate long-side caps (we may add the snapped original for non-Custom)
# - min_short: target minimum short side after scaling (extreme-AR safeguard)
# - global_cap: maximum allowed cap (None = unlimited)
PROFILE_CFG = {
    "highest_quality": {
        "ladder": [1280, 1152, 1088, 1024, 960],
        "min_short": 512,
        "global_cap": 1280,
    },
    "balanced": {
        "ladder": [1024, 960, 896],
        "min_short": 448,
        "global_cap": 1024,  # never exceed 1024
    },
    "speed": {
        "ladder": [896, 832, 768, 704, 640],
        "min_short": 384,
        "global_cap": 896,   # never exceed 896
    },
    "custom": {
        "ladder": [],        # filled with the user target at runtime
        "min_short": 384,
        "global_cap": None,  # no hard cap beyond user input
    },
}

ROUNDING_HELP = (
    "Rounding when snapping to multiples of 16:\n"
    "• nearest: pick the closest multiple of 16\n"
    "• floor: always round down to the next lower multiple of 16\n"
    "• ceil: always round up to the next higher multiple of 16"
)

EXTREME_AR_HELP = (
    "When the image is very tall or wide, the computed short side could get too small. "
    "With this ON, the node will switch to a short-side-first strategy (with sensible floors) "
    "to avoid brittle outputs."
)

CUSTOM_MAX_SIDE_HELP = (
    "Only used when Profile = Custom. Sets the desired long-side cap (divisible by 16). "
    "The node will only downscale toward this value (never upscale)."
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

def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return max(1, a)

def _ratio_str(w: int, h: int) -> str:
    g = _gcd(w, h)
    return f"{w // g}:{h // g}"

def _build_candidates(
    profile_key: str,
    L: int,
    custom_max_side: Optional[int],
    include_snapped_original_for_non_custom: bool,
) -> List[int]:
    cfg = PROFILE_CFG[profile_key]
    ladder = list(cfg["ladder"])

    if profile_key == "custom":
        # For Custom, ONLY use the user target (we will downscale toward it).
        if custom_max_side is None or custom_max_side < 16:
            ladder = [1280]  # safe default if user forgot to set
        else:
            ladder = [int(custom_max_side)]
    else:
        # For non-Custom, we may include the snapped original long side (downscale-only will filter > L).
        if include_snapped_original_for_non_custom:
            ladder.append(_snap16(L, mode="nearest"))

    # Deduplicate and sort descending (prefer larger caps if equally close but ≤ L)
    uniq = sorted({int(x) for x in ladder if x >= 16}, reverse=True)
    return uniq

def _pick_cap_downscale_only(
    profile_key: str,
    L: int,
    rounding_mode: str,
    custom_max_side: Optional[int],
) -> Tuple[int, Dict[str, Any]]:
    """
    Strictly downscale-only picker:
      - Disallow any candidate > L (never upscale)
      - Clamp to profile global cap (if any)
      - For Custom: honor the user target (snapped & clamped to ≤ L)
      - For others: pick the candidate closest to L (but ≤ L and ≤ global cap)
    """
    cfg = PROFILE_CFG[profile_key]
    global_cap = cfg["global_cap"]

    # Build candidates
    candidates = _build_candidates(
        profile_key,
        L,
        custom_max_side,
        include_snapped_original_for_non_custom=True,
    )

    # Helper to clamp to profile global cap
    def clamp_to_cap(x: int) -> int:
        if global_cap is None:
            return x
        return min(x, global_cap)

    # Downscale-only filter: clamp to cap, then reject > L
    filtered: List[int] = []
    for c in candidates:
        c = clamp_to_cap(c)
        if c <= L:
            filtered.append(c)

    # If nothing survived, fall back to min(L, global_cap)
    if not filtered:
        fallback = _snap16(min(L, global_cap if global_cap else L), mode=rounding_mode)
        return fallback, {
            "reason": "fallback_downscale_only",
            "candidates": candidates,
            "filtered": filtered,
        }

    # Custom profile: pick the user target (snapped), not "closest to L"
    if profile_key == "custom":
        target = candidates[0]  # this is the user-specified cap from _build_candidates
        target = clamp_to_cap(target)
        # Enforce downscale-only: cannot exceed L
        target = min(target, L)
        target = _snap16(target, mode=rounding_mode, cap=global_cap)
        return target, {
            "reason": "custom_target_downscale_only",
            "candidates": candidates,
            "filtered": filtered,
        }

    # Non-Custom: choose candidate closest to L (all are ≤ L and ≤ cap)
    chosen = min(filtered, key=lambda c: abs(c - L))
    return chosen, {
        "reason": "closest_to_input_long_side_downscale_only",
        "candidates": candidates,
        "filtered": filtered,
    }

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

    # Pick a long-side target with strict downscale-only logic
    chosen_cap, pick_info = _pick_cap_downscale_only(
        profile_key, L_in, rounding_mode, custom_max_side
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

def _short_side_fallback_targets(profile_key: str) -> List[int]:
    """Short-side-first floors to consider when AR is extreme."""
    cfg = PROFILE_CFG[profile_key]
    base = max(16, int(cfg["min_short"]))
    ladder = sorted({base, base + 64, base + 128, base + 192, base + 256}, reverse=True)
    ladder = [_snap16(x, mode="nearest") for x in ladder]
    ladder = sorted(set(ladder), reverse=True)
    return ladder

# -----------------------------
# ComfyUI Node
# -----------------------------

class WANResolutionHelperNodeV2:
    """
    Inputs:
      - image (IMAGE)
      - profile (CHOICE): Highest Quality / Balanced / Speed / Custom
      - custom_max_side (INT): only used for Custom
      - rounding_mode (CHOICE)
      - extreme_ar_handling (BOOLEAN)
    Outputs:
      - width_out (INT), height_out (INT)
      - aspect_ratio (STRING) : reduced ratio like 3:2, 1:1, 9:16
      - info (STRING)         : detailed summary
      - profile_note (STRING) : short explainer for the chosen profile
    """

    @classmethod
    def INPUT_TYPES(cls):
        profile_labels = [label for key, label in PROFILES]

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image tensor [B,H,W,C]. Only the first in the batch is read."}),
                "profile": (
                    profile_labels,
                    {
                        "default": "Highest Quality (slowest)",
                        "tooltip": (
                            "Choose a profile:\n"
                            "• Highest Quality (slowest): aims near 1280, strictly downscale-only\n"
                            "• Balanced (regular): aims near 1024/960/896 (never above 1024)\n"
                            "• Speed (fastest): aims near 896/832/768... (never above 896)\n"
                            "• Custom: you set the cap; downscale-only"
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
                "extreme_ar_handling": (
                    "BOOLEAN",
                    {"default": True, "tooltip": EXTREME_AR_HELP},
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("width_out", "height_out", "aspect_ratio", "info", "profile_note")
    FUNCTION = "compute"
    CATEGORY = CATEGORY

    _PROFILE_LABEL_TO_KEY = {label: key for key, label in PROFILES}

    def compute(
        self,
        image,
        profile,
        custom_max_side,
        rounding_mode,
        extreme_ar_handling,
    ):
        import torch  # ComfyUI IMAGE is a torch.Tensor

        profile_key = self._PROFILE_LABEL_TO_KEY.get(profile, "balanced")

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
            use_extreme_ar_handling=bool(extreme_ar_handling),
            custom_max_side=int(custom_max_side) if profile_key == "custom" else None,
        )

        profile_note = PROFILE_HELP.get(profile_key, "")
        chosen = info.get("chosen_cap")
        cap16 = info.get("cap16")
        reason = info.get("pick_info", {}).get("reason")

        # Aspect ratio text for the *input* (as requested)
        ar_text = _ratio_str(int(w), int(h))

        summary_lines = [
            f"Profile: {profile}  |  Rounding: {rounding_mode}  |  Downscale-only",
            f"Input: {w}×{h} ({_label_orientation(w,h)}), AR={ar_text}",
        ]
        if profile_key == "custom":
            summary_lines.append(f"Custom target: {custom_max_side}  |  cap16={cap16}")
        else:
            summary_lines.append(f"Chosen cap: {chosen}  (cap16={cap16}, reason={reason})")
        summary_lines.append(f"Output (16x): {out_w}×{out_h} → {_label_orientation(out_w,out_h)}")
        if info.get("used_short_side_fallback"):
            summary_lines.append("Note: Extreme-AR fallback (short-side-first) was used to protect the short side.")
        info_str = "\n".join(summary_lines)

        return (int(out_w), int(out_h), ar_text, info_str, profile_note)


NODE_CLASS_MAPPINGS = {
    "WANResolutionHelperV2": WANResolutionHelperNodeV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WANResolutionHelperV2": "WAN 2.2 Resolution Helper (16x, Profiles)",
}