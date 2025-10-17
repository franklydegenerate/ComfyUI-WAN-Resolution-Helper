# WAN Resolution Helper

# Description: ComfyUI node that outputs WAN 2.2-ready sizes by preserving aspect ratio, capping the long side, and rounding both dimensions to multiples of 16 pixels to reduce artifacts.

# Usage

1. **Add the node**: search for **“WAN 2.2 Resolution Helper (16x, Profiles)”** (category: **WAN/Resolution**).
2. **Connect an IMAGE**: plug any image tensor into the node (it reads the first item in the batch).
3. **Pick a Profile** (or choose **Custom**) to control the *long-side cap*.
4. **Wire outputs**: connect **`width_out`** and **`height_out`** to your **Resize** node.
5. (Optional) Inspect the **`info`** string output—it explains the chosen cap, rounding, and whether extreme-AR fallback was used.

This node **preserves aspect ratio**, **caps the long side**, and **snaps both dimensions to the nearest multiple of 16 pixels** (a common requirement to reduce artifacts and satisfy model constraints).

---

# Node Inputs

### `image` (IMAGE)
Your input image tensor `[B, H, W, C]`. The node reads the first image in the batch to determine the original **width × height**.

### `profile` (Choice)
Preset behavior for the long-side cap. Each profile builds a **ladder** of candidate caps (always 16-aligned), then **chooses the closest sensible cap to your image’s long side** while obeying upscale limits.

- **Highest Quality (slowest)**  
  Aims near **1280** (also considers 1152/1088/1024/960).  
  **Max upscale:** ~**2.0×**; **Minimum short side:** ~**512 px**.  
  Best visual fidelity if VRAM and speed allow.

- **Balanced (regular)**  
  Aims near **1024/960/896**.  
  **Max upscale:** ~**1.5×**; **Minimum short side:** ~**448 px**.  
  Good default for quality/speed balance.

- **Speed (fastest)**  
  Aims near **896/832/768/704/640**.  
  **Max upscale:** ~**1.25×**; **Minimum short side:** ~**384 px**.  
  Great for drafts and quick iterations.

- **Custom (manual)**  
  You provide the cap via **`custom_max_side`**.  
  Can exceed **1280** if your pipeline/VRAM allows. Still preserves aspect ratio and snaps to 16.

> **Notes**
> - Profiles use a **global cap of 1280** by default (except **Custom**).
> - If **Downscale only** is enabled (see `scale_behavior`), the node will not upscale to reach a target cap.
> - The node always adds your **snapped original long side** as a candidate—so if your image is already near a clean 16-multiple, it can keep it.

### `custom_max_side` (INT)
Only used when **profile = Custom**. The desired **long-side cap** (internally snapped to a multiple of 16). Use this if you need **>1280** or a specific cap.

### `rounding_mode` (Choice: `nearest` / `floor` / `ceil`)
How to snap both sides to **multiples of 16**:
- **nearest** (default): closest multiple of 16  
- **floor**: always round down  
- **ceil**: always round up

### `scale_behavior` (Choice: `Scale to target (up/down)` / `Downscale only`)
- **Scale to target (up/down)**: the long side may be up- or down-scaled (within the profile’s **max upscale** limit).  
- **Downscale only**: never upscale; the chosen cap acts as a ceiling.

### `extreme_ar_handling` (BOOLEAN)
When an image is **very tall or very wide**, scaling by long side can make the **short side too small**.  
With this ON (default), the node will **fallback to a short-side-first strategy** using profile-specific minimums (e.g., ~512 for Highest Quality) to keep the output usable. The `info` string notes when this fallback is used.

---

# Node Outputs

### `width_out` (INT), `height_out` (INT)
Final **16-aligned** dimensions that **preserve aspect ratio**, **respect the chosen cap**, and are ready to plug into your **Resize** node.

### `info` (STRING)
Human-readable summary that includes:
- Input size & orientation  
- Selected **profile**, **chosen cap** (and 16-snapped cap), and **why** it was chosen (e.g., “closest to input long side”)  
- Rounding mode & scale behavior  
- Final output size  
- Whether **extreme-AR fallback** was applied

### `profile_note` (STRING)
Concise description of the selected profile’s behavior (useful for UI side panels).

---

# How It Chooses Sizes (Overview)

1. Read original size → determine **long/short side** and aspect ratio.  
2. Build **candidate caps** (profile ladder + snapped original long side).  
3. Filter by global cap (usually ≤1280 unless **Custom**), **upscale limit** (profile-specific), and **scale behavior**.  
4. Pick the **closest candidate** to the original long side.  
5. Compute the other side from the **aspect ratio**, then **snap both to multiples of 16**.  
6. If the short side violates the profile **minimum** (extreme AR), switch to **short-side-first** and try a small ladder of short-side targets to meet that minimum; then recompute the long side and snap again.

---

# Examples

- **1706×2714 (portrait)**  
  - **Highest Quality:** ~**1280×800** (snapped)  
  - **Balanced:** ~**1024×640**  
  - **Speed:** ~**768×480**

- **960×960 (square)**  
  - **Balanced:** stays **960×960** (already clean and near target)  
  - **Highest Quality:** may go to **1024×1024** if up/down scaling is allowed and within upscale limit

- **640×1920 (very tall)**  
  - Long-side-first may yield too-small short side → **fallback** to short-side-first to keep short ≥ profile minimum (see `info`).

---

# Tips

- If a downstream node needs a specific cap (e.g., **1280**), use **Highest Quality** or **Custom** with that value.  
- For strict no-upscale pipelines, set **`scale_behavior` = Downscale only**.  
- If you get a very thin output on extreme panoramas/tall shots, keep **`extreme_ar_handling`** ON.  
- To experiment above **1280**, use **Custom** (ensure your VRAM/pipeline supports it).
