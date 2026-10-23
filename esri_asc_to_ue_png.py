#!/usr/bin/env python3
"""
esri_asc_to_ue_png.py

Convert an ESRI ASCII Grid heightmap (.asc) into a UE5-compatible
16-bit grayscale PNG heightmap.

Input header example:

    ncols 1600
    nrows 1000
    xllcenter 6533600.25
    yllcenter 6037000.25
    cellsize 0.50
    nodata_value -9999

Behavior:
  - nodata_value is ignored when computing min/max.
  - nodata_value cells are replaced with the minimum valid height.
  - padded/outside cells are also filled with the minimum valid height.
  - therefore nodata / padded cells become raw value 0 in the PNG.
  - UE Landscape X/Y scale is calculated from cellsize.
  - UE Landscape Z scale/location are calculated so heights preserve meters.
  - Output is a 16-bit grayscale PNG (no alpha, no color profile).

Dependencies:
    pip install numpy pillow
"""

import argparse
import json
import sys

import numpy as np
from PIL import Image


ALLOWED_SECTION_SIZES = (255, 127, 63, 31, 15, 7)

KNOWN_HEADER_KEYS = {
    "ncols",
    "nrows",
    "xllcenter",
    "yllcenter",
    "xllcorner",
    "yllcorner",
    "cellsize",
    "nodata_value",
}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# ---------------------------------------------------------------------
# Reading ESRI ASCII grid
# ---------------------------------------------------------------------

def parse_float_line(line, expected_count):
    """Parse one whitespace-separated float line."""
    row = np.fromstring(line, dtype=np.float32, sep=" ")

    if row.size != expected_count:
        row = np.array(line.split(), dtype=np.float32)

    if row.size != expected_count:
        raise ValueError(
            f"Expected {expected_count} values, got {row.size} in line: "
            f"{line[:80]}..."
        )

    return row


def read_esri_asc(path):
    """Read ESRI ASCII grid. Returns (data, meta)."""
    header = {}

    with open(path, "r") as f:
        line = f.readline()

        while line:
            stripped = line.strip()
            if not stripped:
                line = f.readline()
                continue

            parts = stripped.split()
            key = parts[0].lower()

            if key in KNOWN_HEADER_KEYS and len(parts) >= 2:
                header[key] = parts[1]
                line = f.readline()
            else:
                break

        for key in ("ncols", "nrows", "cellsize"):
            if key not in header:
                raise ValueError(f"Missing required header key: {key}")

        ncols = int(header["ncols"])
        nrows = int(header["nrows"])
        cellsize = float(header["cellsize"])

        if ncols <= 0 or nrows <= 0:
            raise ValueError("ncols and nrows must be positive.")
        if cellsize <= 0.0:
            raise ValueError("cellsize must be positive.")

        if "xllcenter" in header:
            xllcenter = float(header["xllcenter"])
        elif "xllcorner" in header:
            xllcenter = float(header["xllcorner"]) + cellsize * 0.5
        else:
            raise ValueError("Missing xllcenter / xllcorner.")

        if "yllcenter" in header:
            yllcenter = float(header["yllcenter"])
        elif "yllcorner" in header:
            yllcenter = float(header["yllcorner"]) + cellsize * 0.5
        else:
            raise ValueError("Missing yllcenter / yllcorner.")

        nodata_value = None
        if "nodata_value" in header:
            try:
                nodata_value = float(header["nodata_value"])
            except ValueError:
                nodata_value = None

        data = np.empty((nrows, ncols), dtype=np.float32)

        for y in range(nrows):
            while line and line.strip() == "":
                line = f.readline()

            if not line:
                raise EOFError(
                    f"Unexpected end of file at row {y + 1}. "
                    f"Expected {nrows} rows."
                )

            row = parse_float_line(line, ncols)
            data[y, :] = row

            if (y + 1) % 100 == 0:
                eprint(f"Read {y + 1}/{nrows} rows...")

            line = f.readline()

    meta = {
        "ncols": ncols,
        "nrows": nrows,
        "cellsize": cellsize,
        "xllcenter": xllcenter,
        "yllcenter": yllcenter,
        "nodata_value": nodata_value,
    }

    return data, meta


# ---------------------------------------------------------------------
# UE-friendly size selection
# ---------------------------------------------------------------------

def choose_target_size(width, height, section_size="auto", max_components=1024):
    """
    Choose target = components * section_size + 1, larger than source.
    """
    if section_size == "auto":
        sizes = ALLOWED_SECTION_SIZES
    else:
        section_size = int(section_size)
        if section_size not in ALLOWED_SECTION_SIZES:
            raise ValueError(
                f"Section size must be one of {ALLOWED_SECTION_SIZES}, or 'auto'."
            )
        sizes = (section_size,)

    candidates = []

    for s in sizes:
        cx = max(1, (width - 1 + s - 1) // s)
        cy = max(1, (height - 1 + s - 1) // s)

        tw = cx * s + 1
        th = cy * s + 1

        candidates.append(
            {
                "section_size": s,
                "components_x": cx,
                "components_y": cy,
                "components": cx * cy,
                "target_width": tw,
                "target_height": th,
                "area": tw * th,
            }
        )

    if section_size != "auto":
        return candidates[0]

    feasible = [c for c in candidates if c["components"] <= max_components]

    if feasible:
        feasible.sort(
            key=lambda c: (c["area"], c["components"], -c["section_size"])
        )
        return feasible[0]

    candidates.sort(
        key=lambda c: (c["components"], c["area"], -c["section_size"])
    )

    chosen = candidates[0]
    eprint(
        f"Warning: no size under --max-components={max_components}. "
        f"Using section size {chosen['section_size']} "
        f"with {chosen['components']} components."
    )
    return chosen


def place_on_canvas(src, target_w, target_h, anchor, pad_value):
    """Place source into a larger canvas filled with pad_value."""
    src_h, src_w = src.shape

    if target_w < src_w or target_h < src_h:
        raise ValueError("Target size is smaller than source data.")

    canvas = np.full((target_h, target_w), pad_value, dtype=np.float32)

    if anchor == "center":
        x = (target_w - src_w) // 2
        y = (target_h - src_h) // 2
    elif anchor == "bottom-left":
        x = 0
        y = target_h - src_h
    elif anchor == "bottom-right":
        x = target_w - src_w
        y = target_h - src_h
    else:  # top-left
        x = 0
        y = 0

    canvas[y:y + src_h, x:x + src_w] = src
    return canvas, x, y


# ---------------------------------------------------------------------
# Geographic placement
# ---------------------------------------------------------------------

def compute_geographic_info(meta, src_w, src_h, target_w, target_h,
                            offset_x, offset_y):
    """
    Assumes xllcenter/yllcenter are lower-left cell centers,
    ASC rows top->bottom, cellsize in meters.
    """
    cellsize = meta["cellsize"]

    src_x_min = meta["xllcenter"]
    src_y_min = meta["yllcenter"]

    src_x_max = src_x_min + (src_w - 1) * cellsize
    src_y_max = src_y_min + (src_h - 1) * cellsize

    target_x_min = src_x_min - offset_x * cellsize
    target_y_max = src_y_max + offset_y * cellsize

    target_x_max = target_x_min + (target_w - 1) * cellsize
    target_y_min = target_y_max - (target_h - 1) * cellsize

    return {
        "source_x_min_center_m": src_x_min,
        "source_y_min_center_m": src_y_min,
        "source_x_max_center_m": src_x_max,
        "source_y_max_center_m": src_y_max,
        "source_center_x_m": src_x_min + (src_w - 1) * cellsize * 0.5,
        "source_center_y_m": src_y_min + (src_h - 1) * cellsize * 0.5,
        "target_x_min_center_m": target_x_min,
        "target_y_min_center_m": target_y_min,
        "target_x_max_center_m": target_x_max,
        "target_y_max_center_m": target_y_max,
        "target_center_x_m": target_x_min + (target_w - 1) * cellsize * 0.5,
        "target_center_y_m": target_y_min + (target_h - 1) * cellsize * 0.5,
    }


# ---------------------------------------------------------------------
# PNG writing (16-bit grayscale, verified)
# ---------------------------------------------------------------------

def save_16bit_png(raw_u16, path):
    """
    Write a true 16-bit grayscale PNG and verify it.

    - no alpha
    - no palette
    - no ICC profile
    - no gamma correction
    - values written exactly as uint16
    """
    h, w = raw_u16.shape
    arr = np.ascontiguousarray(raw_u16, dtype=np.uint16)

    # Pillow maps uint16 arrays to mode "I;16" on current versions.
    img = Image.fromarray(arr)

    # Fallback for older Pillow versions that may map uint16 to mode "I".
    if img.mode not in ("I;16", "I;16L", "I;16B"):
        img = Image.frombytes(
            "I;16",
            (w, h),
            arr.astype("<u2").tobytes(),
        )

    # Save without color management or palette conversion.
    img.save(path, format="PNG", optimize=False)

    # Verify that the file really is 16-bit grayscale and values survived.
    with Image.open(path) as check:
        if check.mode not in ("I;16", "I;16L", "I;16B", "I"):
            raise RuntimeError(
                f"PNG was not written as 16-bit grayscale. "
                f"Got mode={check.mode}"
            )

        if check.size != (w, h):
            raise RuntimeError(
                f"PNG size mismatch: expected {(w, h)}, got {check.size}"
            )

        # Spot-check corners and a center pixel.
        for (yy, xx) in ((0, 0), (0, w - 1), (h - 1, 0),
                         (h - 1, w - 1), (h // 2, w // 2)):
            expected = int(arr[yy, xx])
            actual = int(check.getpixel((xx, yy)))
            if actual != expected:
                raise RuntimeError(
                    f"PNG value mismatch at ({xx},{yy}): "
                    f"expected {expected}, got {actual}"
                )

    return {
        "png_width": w,
        "png_height": h,
        "png_mode": "16-bit grayscale",
        "png_bit_depth": 16,
        "png_has_alpha": False,
        "png_verified": True,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert ESRI ASCII heightmap to UE5 16-bit grayscale PNG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input", help="Input ESRI .asc heightmap file")
    parser.add_argument("output", help="Output .png file (16-bit grayscale)")

    parser.add_argument(
        "--section-size", default="auto",
        help="UE section size: auto or one of 7,15,31,63,127,255",
    )
    parser.add_argument(
        "--max-components", type=int, default=1024,
        help="Component budget when --section-size auto",
    )
    parser.add_argument(
        "--anchor",
        choices=["top-left", "center", "bottom-left", "bottom-right"],
        default="bottom-left",
        help="Placement of source data inside padded canvas",
    )
    parser.add_argument(
        "--flip-vertical", action="store_true",
        help="Flip final image vertically before saving",
    )
    parser.add_argument(
        "--flip-horizontal", action="store_true",
        help="Flip final image horizontally before saving",
    )
    parser.add_argument(
        "--origin-mode", choices=["absolute", "center", "lower-left"],
        default="center",
        help="World origin reference for UE X/Y location",
    )
    parser.add_argument(
        "--origin-x", type=float, default=None,
        help="Custom GIS X origin in meters (overrides --origin-mode)",
    )
    parser.add_argument(
        "--origin-y", type=float, default=None,
        help="Custom GIS Y origin in meters (overrides --origin-mode)",
    )
    parser.add_argument(
        "--axis-convention", choices=["xy", "yx"], default="xy",
        help="xy: GIS X->UE X, GIS Y->UE Y. yx: swapped.",
    )
    parser.add_argument(
        "--invert-x", action="store_true",
        help="Invert calculated UE X location",
    )
    parser.add_argument(
        "--invert-y", action="store_true",
        help="Invert calculated UE Y location",
    )
    parser.add_argument(
        "--also-raw", default=None,
        help="Optional: also write a .r16 for comparison/debugging",
    )
    parser.add_argument(
        "--info-file", default=None,
        help="Optional path to write JSON metadata",
    )

    args = parser.parse_args()

    if not args.output.lower().endswith(".png"):
        eprint("Warning: output extension is not .png; writing PNG anyway.")

    # -----------------------------------------------------------------
    # 1. Read ASC
    # -----------------------------------------------------------------
    data, meta = read_esri_asc(args.input)

    nrows = meta["nrows"]
    ncols = meta["ncols"]
    cellsize = meta["cellsize"]
    nodata_value = meta["nodata_value"]

    # -----------------------------------------------------------------
    # 2. Nodata handling: nodata -> min valid -> raw 0
    # -----------------------------------------------------------------
    finite_mask = np.isfinite(data)

    if nodata_value is not None and np.isfinite(nodata_value):
        valid_mask = finite_mask & (data != nodata_value)
    else:
        valid_mask = finite_mask

    valid_count = int(np.count_nonzero(valid_mask))
    invalid_count = int(data.size - valid_count)

    if valid_count > 0:
        valid_values = data[valid_mask]
        min_valid = float(np.min(valid_values))
        max_valid = float(np.max(valid_values))
    else:
        eprint("Warning: no valid height values. Using flat zero terrain.")
        min_valid = 0.0
        max_valid = 0.0

    if invalid_count:
        data[~valid_mask] = min_valid
        eprint(
            f"Replaced {invalid_count} nodata/invalid cells "
            f"with minimum valid height {min_valid}."
        )

    # -----------------------------------------------------------------
    # 3. Choose larger UE-friendly size
    # -----------------------------------------------------------------
    target = choose_target_size(
        ncols, nrows,
        section_size=args.section_size,
        max_components=args.max_components,
    )

    target_w = target["target_width"]
    target_h = target["target_height"]

    # -----------------------------------------------------------------
    # 4. Pad with min valid height (so padding -> raw 0)
    # -----------------------------------------------------------------
    canvas, offset_x, offset_y = place_on_canvas(
        data, target_w, target_h, args.anchor, min_valid
    )

    # -----------------------------------------------------------------
    # 5. Geographic info (computed before any flip)
    # -----------------------------------------------------------------
    geo = compute_geographic_info(
        meta, ncols, nrows, target_w, target_h, offset_x, offset_y
    )

    # -----------------------------------------------------------------
    # 6. Optional orientation fix
    # -----------------------------------------------------------------
    if args.flip_vertical:
        canvas = np.flipud(canvas)
    if args.flip_horizontal:
        canvas = np.fliplr(canvas)

    canvas = np.ascontiguousarray(canvas)

    # -----------------------------------------------------------------
    # 7. Encode heights to uint16
    #
    # min_valid -> 0
    # max_valid -> 65535
    #
    # UE height model:
    #   WorldZ_cm = LandscapeZ_cm + (Raw - 32768) * (ScaleZ / 128)
    # -----------------------------------------------------------------
    if max_valid <= min_valid:
        # Flat terrain: use a synthetic 1 m range so raw 0 maps to min_valid.
        range_min = min_valid
        height_range_m = 0.0

        range_cm = 1.0 * 100.0
        z_scale = range_cm * 128.0 / 65535.0
        z_position_cm = range_min * 100.0 + 32768.0 * z_scale / 128.0

        raw = np.zeros(canvas.shape, dtype=np.uint16)

        height_note = (
            "Flat input. Synthetic 1 m range used so raw 0 maps to "
            "the flat height."
        )
    else:
        range_min = min_valid
        height_range_m = max_valid - min_valid

        range_cm = height_range_m * 100.0
        z_scale = range_cm * 128.0 / 65535.0
        z_position_cm = range_min * 100.0 + 32768.0 * z_scale / 128.0

        norm = canvas.astype(np.float64)
        norm -= range_min
        norm /= height_range_m
        np.clip(norm, 0.0, 1.0, out=norm)
        norm *= 65535.0
        np.rint(norm, out=norm)

        raw = norm.astype(np.uint16)

        height_note = (
            "Raw 0 = minimum valid height. "
            "Raw 65535 = maximum valid height. "
            "nodata/padding therefore become raw 0."
        )

    # -----------------------------------------------------------------
    # 8. Write 16-bit grayscale PNG
    # -----------------------------------------------------------------
    png_info = save_16bit_png(raw, args.output)

    # Optional raw sidecar for debugging.
    if args.also_raw:
        raw.astype("<u2").tofile(args.also_raw)
        eprint(f"Also wrote raw: {args.also_raw}")

    # -----------------------------------------------------------------
    # 9. UE X/Y scale and location
    # -----------------------------------------------------------------
    xy_scale = cellsize * 100.0  # meters -> cm

    if args.origin_x is not None or args.origin_y is not None:
        origin_x = args.origin_x if args.origin_x is not None else 0.0
        origin_y = args.origin_y if args.origin_y is not None else 0.0
        origin_mode = "custom"
    else:
        origin_mode = args.origin_mode
        if args.origin_mode == "absolute":
            origin_x, origin_y = 0.0, 0.0
        elif args.origin_mode == "lower-left":
            origin_x = geo["target_x_min_center_m"]
            origin_y = geo["target_y_min_center_m"]
        else:  # center
            origin_x = geo["target_center_x_m"]
            origin_y = geo["target_center_y_m"]

    dx = geo["target_center_x_m"] - origin_x
    dy = geo["target_center_y_m"] - origin_y

    if args.axis_convention == "xy":
        loc_x_m, loc_y_m = dx, dy
    else:
        loc_x_m, loc_y_m = dy, dx

    if args.invert_x:
        loc_x_m = -loc_x_m
    if args.invert_y:
        loc_y_m = -loc_y_m

    # -----------------------------------------------------------------
    # 10. Report
    # -----------------------------------------------------------------
    info = {
        "input": args.input,
        "output": args.output,
        "output_format": "16-bit grayscale PNG",
        "source_width": ncols,
        "source_height": nrows,
        "target_width": target_w,
        "target_height": target_h,
        "section_size": target["section_size"],
        "components_x": target["components_x"],
        "components_y": target["components_y"],
        "components_total": target["components"],
        "anchor": args.anchor,
        "flip_vertical": bool(args.flip_vertical),
        "flip_horizontal": bool(args.flip_horizontal),
        "cellsize_m": cellsize,
        "nodata_value": nodata_value,
        "valid_cells": valid_count,
        "nodata_or_invalid_cells": invalid_count,
        "nodata_replaced_with_m": min_valid,
        "height_min_m": min_valid,
        "height_max_m": max_valid,
        "height_range_m": height_range_m,
        "height_note": height_note,
        "landscape_scale_x": xy_scale,
        "landscape_scale_y": xy_scale,
        "landscape_scale_z": z_scale,
        "landscape_location_x_cm": loc_x_m * 100.0,
        "landscape_location_y_cm": loc_y_m * 100.0,
        "landscape_location_z_cm": z_position_cm,
        "landscape_location_z_m": z_position_cm / 100.0,
        "axis_convention": args.axis_convention,
        "origin_mode": origin_mode,
        "origin_x_m": origin_x,
        "origin_y_m": origin_y,
        "gis": geo,
        "png": png_info,
        "ue_notes": (
            "Import the PNG into Landscape. Ensure it is imported as a "
            "16-bit grayscale heightmap (not 8-bit). "
            "Set Landscape Scale X/Y/Z and Location X/Y/Z to the printed "
            "values. Default origin-mode=center keeps the landscape near "
            "world origin. If upside down, rerun with --flip-vertical. "
            "If north/east are swapped, try --axis-convention yx."
        ),
    }

    print(json.dumps(info, indent=2))

    if args.info_file:
        with open(args.info_file, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

    eprint(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
    
