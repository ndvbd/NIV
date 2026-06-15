from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter

from fontTools.ttLib import TTFont


TARGET_UPM = 2048


GLOBAL_MAX_DIFF_PHANTOM_TO_OUTLINE = -(10**9)
GLOBAL_MAX_DIFF_WHERE = ("", "")

GLOBAL_AXIS_COORD_COUNTS = Counter()
GLOBAL_AXIS_COORD_SEEN_COUNTS = Counter()
GLOBAL_FVAR_AXIS_COUNTS = Counter()
GLOBAL_UNICODE_SEEN_COUNTS = Counter()
GLOBAL_UNICODE_USED_COUNTS = Counter()
GLOBAL_TUPLE_NUM_AXES_COUNTS = Counter()
GLOBAL_FONT_NUM_AXES_COUNTS = Counter()
GLOBAL_TUPLES_WITH_UNUSUAL_PEAK = 0
GLOBAL_TUPLES_DEFAULT_TENT = 0
GLOBAL_TUPLES_WITH_EXPLICIT_TENT = 0
GLOBAL_TUPLES_WITH_EXPLICIT_MIN = 0
GLOBAL_TUPLES_WITH_EXPLICIT_MAX = 0
GLOBAL_TUPLES_WITH_EXPLICIT_BOTH = 0


def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def split_contours(points: List[dict], end_pts: List[int]) -> List[List[dict]]:
    contours: List[List[dict]] = []
    start = 0
    for end in end_pts:
        contours.append(points[start : end + 1])
        start = end + 1
    return contours


def _glyph_has_any_gvar(tt: TTFont, gname: str) -> bool:
    if "gvar" not in tt:
        return False
    gvar = tt["gvar"]
    variations = gvar.variations.get(gname)
    return bool(variations)


def _component_transform_string(c) -> str:

    if hasattr(c, "transform"):
        return str(getattr(c, "transform"))

    x = getattr(c, "x", None)
    y = getattr(c, "y", None)
    if x is not None or y is not None:
        return f"translate({int(x or 0)},{int(y or 0)})"

    for meth in ("getComponentTransformation", "getComponentInfo"):
        if hasattr(c, meth):
            try:
                return str(getattr(c, meth)())
            except Exception:
                pass

    return "unknown"


def _scale_int(v: int, s: float) -> int:

    return int(round(v * s))


def _max_delta_pt_plus_one(tt: TTFont, gname: str) -> int:

    if "gvar" not in tt:
        return 0
    gvar = tt["gvar"]
    variations = gvar.variations.get(gname)
    if not variations:
        return 0

    max_pt = -1
    for var in variations:
        coords = getattr(var, "coordinates", None) or []
        for i, d in enumerate(coords):
            if d is None:
                continue
            dx, dy = int(d[0]), int(d[1])
            if dx == 0 and dy == 0:
                continue
            if i > max_pt:
                max_pt = i

    return max_pt + 1


def _build_glyph_to_codepoints(tt: TTFont) -> Dict[str, List[int]]:

    best = tt.getBestCmap() or {}
    g2cps: Dict[str, List[int]] = {}
    for cp, gname in best.items():
        g2cps.setdefault(gname, []).append(int(cp))
    for gname in list(g2cps.keys()):
        g2cps[gname] = sorted(set(g2cps[gname]))
    return g2cps


def _is_simple_ascii_alnum_codepoint(cp: int) -> bool:

    if 0x30 <= cp <= 0x39:
        return True

    if 0x41 <= cp <= 0x5A:
        return True

    if 0x61 <= cp <= 0x7A:
        return True
    return False


def _glyph_passes_only_simple_glyphs_filter(codepoints: List[int]) -> bool:

    return any(_is_simple_ascii_alnum_codepoint(cp) for cp in codepoints)


def _format_codepoints(codepoints: List[int]) -> str:

    return ",".join(f"U+{cp:04X}" for cp in codepoints)


def _safe_char_for_xml(cp: int) -> str:
    try:
        ch = chr(cp)
    except Exception:
        return ""

    if cp < 0x20:
        return ""
    return esc(ch)


def glyph_to_ttx_like(
    tt: TTFont,
    gname: str,
    indent: str = "  ",
    skip_composites: bool = False,
    upm_scale: float = 1.0,
    axes_used: Optional[Set[str]] = None,
) -> Optional[str]:
    glyf = tt["glyf"]
    if gname not in glyf.glyphs:
        return None

    g = glyf[gname]
    if g.isComposite():
        if skip_composites:
            return None

        comps = []
        for c in g.components:
            tr = _component_transform_string(c)
            comps.append(
                f'{indent}<component glyphName="{esc(c.glyphName)}" '
                f'transform="{esc(tr)}"/>'
            )
        ttglyph = (
            f'<TTGlyph name="{esc(gname)}">\n'
            + "\n".join(comps)
            + ("\n" if comps else "")
            + f"</TTGlyph>"
        )
        return ttglyph

    coords, end_pts, flags = g.getCoordinates(glyf)
    points = [
        {
            "x": _scale_int(int(x), upm_scale),
            "y": _scale_int(int(y), upm_scale),
            "on": 1 if (f & 1) else 0,
        }
        for (x, y), f in zip(coords, flags)
    ]
    contours = split_contours(points, list(map(int, end_pts)))

    axes_used = axes_used or set()
    axes_text = ",".join(sorted(axes_used))

    lines: List[str] = []
    lines.append(f'<TTGlyph name="{esc(gname)}">')
    lines.append(f"{indent}<axes>{esc(axes_text)}</axes>")
    for contour in contours:
        lines.append(f"{indent}<contour>")
        for pt in contour:
            lines.append(
                f'{indent}{indent}<pt x="{pt["x"]}" y="{pt["y"]}" on="{pt["on"]}"/>'
            )
        lines.append(f"{indent}</contour>")
    lines.append(f"</TTGlyph>")
    return "\n".join(lines)


def _as_peak_scalar(v) -> float:
    if isinstance(v, (tuple, list)):
        if len(v) >= 2:
            return float(v[1])
        if len(v) == 1:
            return float(v[0])
        return 0.0
    return float(v)


def _is_usual_peak_value(v: float, eps: float = 1e-9) -> bool:
    return abs(v + 1.0) < eps or abs(v) < eps or abs(v - 1.0) < eps


def _infer_1d_iup(t: float, a: float, b: float, da: float, db: float) -> float:

    if a == b:
        return da if da == db else 0.0

    lo = a if a < b else b
    hi = b if a < b else a
    if t <= lo:

        return da if a == lo else db
    if t >= hi:

        return da if a == hi else db

    p = (t - a) / (b - a)
    return da + p * (db - da)


def _infer_gvar_missing_deltas_for_glyph(
    tt: TTFont,
    gname: str,
    coords_default_xy: List[Tuple[float, float]],
    end_pts: List[int],
    var_coords: List[Optional[Tuple[float, float]]],
) -> List[Tuple[float, float]]:

    glyf = tt["glyf"]
    g = glyf[gname]

    n_total = len(var_coords)
    out: List[Tuple[float, float]] = [(0.0, 0.0)] * n_total
    for i, d in enumerate(var_coords):
        if d is None:
            continue
        out[i] = (float(d[0]), float(d[1]))

    if g.isComposite() or not end_pts:
        return out

    outline_n = int(end_pts[-1]) + 1
    outline_n = min(outline_n, n_total)

    contours: List[List[int]] = []
    start = 0
    for e in end_pts:
        e = int(e)
        if e < start:
            continue
        contours.append(list(range(start, e + 1)))
        start = e + 1

    for contour in contours:
        contour = [i for i in contour if i < outline_n]
        if not contour:
            continue

        referenced = [i for i in contour if var_coords[i] is not None]
        if len(referenced) == 0:

            continue

        if len(referenced) == 1:

            r = referenced[0]
            dx, dy = out[r]
            for i in contour:
                if var_coords[i] is None:
                    out[i] = (dx, dy)
            continue

        ref_set = set(referenced)

        pos_of = {idx: p for p, idx in enumerate(contour)}
        ref_positions = sorted(pos_of[i] for i in referenced)

        for idx in contour:
            if idx in ref_set:
                continue

            p = pos_of[idx]

            prev_p = None
            for rp in reversed(ref_positions):
                if rp < p:
                    prev_p = rp
                    break
            if prev_p is None:
                prev_p = ref_positions[-1]

            next_p = None
            for rp in ref_positions:
                if rp > p:
                    next_p = rp
                    break
            if next_p is None:
                next_p = ref_positions[0]

            prev_i = contour[prev_p]
            next_i = contour[next_p]

            tx, ty = coords_default_xy[idx]
            ax, ay = coords_default_xy[prev_i]
            bx, by = coords_default_xy[next_i]
            dax, day = out[prev_i]
            dbx, dby = out[next_i]

            ix = _infer_1d_iup(float(tx), float(ax), float(bx), float(dax), float(dbx))
            iy = _infer_1d_iup(float(ty), float(ay), float(by), float(day), float(dby))
            out[idx] = (ix, iy)

    return out


def gvar_to_ttx_like(
    font_path: Path,
    tt: TTFont,
    gname: str,
    indent: str,
    upm_scale: float,
    only_axes: Optional[Set[str]],
    infer_missing_deltas: bool,
) -> Optional[Tuple[str, Set[str]]]:
    global GLOBAL_TUPLES_WITH_UNUSUAL_PEAK
    global GLOBAL_TUPLES_DEFAULT_TENT, GLOBAL_TUPLES_WITH_EXPLICIT_TENT
    global GLOBAL_TUPLES_WITH_EXPLICIT_MIN, GLOBAL_TUPLES_WITH_EXPLICIT_MAX, GLOBAL_TUPLES_WITH_EXPLICIT_BOTH

    if "gvar" not in tt:
        return None

    gvar = tt["gvar"]
    variations = gvar.variations.get(gname)
    if not variations:
        return None

    used_axes_for_glyph: Set[str] = set()

    lines: List[str] = []
    lines.append(f'<glyphVariations glyph="{esc(gname)}">')

    kept_any_tuple = False

    glyf = tt["glyf"]
    coords_default_xy: List[Tuple[float, float]] = []
    end_pts: List[int] = []
    try:
        g = glyf[gname]
        if not g.isComposite():

            coords_def, end_pts_def, _flags_def = g.getCoordinates(glyf)

            coords_default_xy = [(float(x), float(y)) for (x, y) in coords_def]

            end_pts = list(map(int, end_pts_def))
        else:
            return None
    except Exception:
        coords_default_xy = []
        end_pts = []

    for var in variations:
        support = getattr(var, "support", None) or {}
        axes = set()
        if support:
            axes.update(support.keys())
        axes_attr = getattr(var, "axes", None) or {}
        if isinstance(axes_attr, dict):
            axes.update(axes_attr.keys())

        tuple_axes = set(axes)

        for axis_tag in tuple_axes:
            GLOBAL_AXIS_COORD_SEEN_COUNTS[axis_tag] += 1

        if only_axes is not None and len(only_axes) > 0:
            if not tuple_axes.issubset(only_axes):
                continue

        kept_any_tuple = True
        GLOBAL_TUPLE_NUM_AXES_COUNTS[len(tuple_axes)] += 1
        tuple_has_unusual_peak = False
        tuple_has_explicit_tent = False
        tuple_has_explicit_min = False
        tuple_has_explicit_max = False
        tuple_has_explicit_both = False
        lines.append(f"{indent}<tuple>")

        for axis_tag in sorted(axes):

            min_val = None
            peak_val = None
            max_val = None

            if axis_tag in support:
                sup = support[axis_tag]
                if isinstance(sup, dict):
                    allowed_support_keys = {"min", "peak", "default", "max"}
                    bad_keys = set(sup.keys()) - allowed_support_keys
                    if bad_keys:
                        raise RuntimeError(
                            f"Unexpected support dict keys in gvar tuple for font={font_path.name}, "
                            f"glyph={gname}, axis={axis_tag}: got keys={sorted(sup.keys())}, "
                            f"unknown={sorted(bad_keys)}. Allowed keys are "
                            f"{sorted(allowed_support_keys)}."
                        )
                    if "peak" not in sup and "default" not in sup:
                        raise RuntimeError(
                            f"Support dict missing peak/default in gvar tuple for font={font_path.name}, "
                            f"glyph={gname}, axis={axis_tag}: got keys={sorted(sup.keys())}."
                        )

                    min_val = (
                        float(sup["min"])
                        if "min" in sup and sup["min"] is not None
                        else None
                    )
                    peak_raw = sup["peak"] if "peak" in sup else sup.get("default")
                    if peak_raw is None:
                        raise RuntimeError(
                            f"Support dict has null peak/default in gvar tuple for font={font_path.name}, "
                            f"glyph={gname}, axis={axis_tag}."
                        )
                    peak_val = float(peak_raw)
                    max_val = (
                        float(sup["max"])
                        if "max" in sup and sup["max"] is not None
                        else None
                    )
                elif isinstance(sup, (tuple, list)):

                    if len(sup) == 3:
                        min_val, peak_val, max_val = (
                            float(sup[0]),
                            float(sup[1]),
                            float(sup[2]),
                        )
                    elif len(sup) == 1:
                        peak_val = float(sup[0])
                    else:
                        raise RuntimeError(
                            f"Unexpected support tuple/list length in gvar tuple for font={font_path.name}, "
                            f"glyph={gname}, axis={axis_tag}: len={len(sup)}, value={sup!r}. "
                            "Expected len=1 (peak-only) or len=3 (min,peak,max)."
                        )
                else:

                    try:
                        peak_val = float(sup)
                    except Exception as e:
                        raise RuntimeError(
                            f"Unsupported support value type in gvar tuple for font={font_path.name}, "
                            f"glyph={gname}, axis={axis_tag}: type={type(sup).__name__}, value={sup!r}"
                        ) from e
            else:
                peak_val = _as_peak_scalar(axes_attr[axis_tag])

            coord_parts = [f'axis="{esc(axis_tag)}"']
            if min_val is not None:
                coord_parts.append(f'min="{float(min_val):.6g}"')
                tuple_has_explicit_min = True
                tuple_has_explicit_tent = True
            coord_parts.append(f'value="{float(peak_val):.6g}"')
            if max_val is not None:
                coord_parts.append(f'max="{float(max_val):.6g}"')
                tuple_has_explicit_max = True
                tuple_has_explicit_tent = True
            if min_val is not None and max_val is not None:
                tuple_has_explicit_both = True

            coord_line = f'{indent}{indent}<coord {" ".join(coord_parts)}/>'
            lines.append(coord_line)
            GLOBAL_AXIS_COORD_COUNTS[axis_tag] += 1
            used_axes_for_glyph.add(axis_tag)
            if not _is_usual_peak_value(float(peak_val)):
                tuple_has_unusual_peak = True

        if tuple_has_unusual_peak:
            GLOBAL_TUPLES_WITH_UNUSUAL_PEAK += 1
        if tuple_has_explicit_tent:
            GLOBAL_TUPLES_WITH_EXPLICIT_TENT += 1
        else:
            GLOBAL_TUPLES_DEFAULT_TENT += 1
        if tuple_has_explicit_min:
            GLOBAL_TUPLES_WITH_EXPLICIT_MIN += 1
        if tuple_has_explicit_max:
            GLOBAL_TUPLES_WITH_EXPLICIT_MAX += 1
        if tuple_has_explicit_both:
            GLOBAL_TUPLES_WITH_EXPLICIT_BOTH += 1

        coords = getattr(var, "coordinates", None) or []

        if infer_missing_deltas and len(coords_default_xy) + 4 == len(coords):

            filled = _infer_gvar_missing_deltas_for_glyph(
                tt=tt,
                gname=gname,
                coords_default_xy=coords_default_xy,
                end_pts=end_pts,
                var_coords=[
                    (float(d[0]), float(d[1])) if d is not None else None
                    for d in coords
                ],
            )

            for i, (dx_f, dy_f) in enumerate(filled):

                dx = int(round(dx_f * upm_scale))
                dy = int(round(dy_f * upm_scale))

                what_to_append = f'{indent}{indent}<delta pt="{i}" x="{dx}" y="{dy}"/>'

                lines.append(what_to_append)

        else:
            mes = "** This was the old behavior before missing deltas were inferred. (exit) [protection]"
            print(mes)
            exit(1)

            for i, d in enumerate(coords):
                if d is None:
                    continue

                dx, dy = int(d[0]), int(d[1])
                dx = _scale_int(dx, upm_scale)
                dy = _scale_int(dy, upm_scale)

                lines.append(f'{indent}{indent}<delta pt="{i}" x="{dx}" y="{dy}"/>')

        lines.append(f"{indent}</tuple>")

    lines.append(f"</glyphVariations>")

    if not kept_any_tuple:
        return None

    return "\n".join(lines), used_axes_for_glyph


def extract_font_ttx_like(
    font_path: Path,
    out_path: Path,
    *,
    gzip_output: bool,
    skip_composites: bool,
    upm_scale: float,
    only_simple_glyphs: bool,
    only_axes: Set[str],
    infer_missing_deltas: bool,
) -> None:
    global GLOBAL_MAX_DIFF_PHANTOM_TO_OUTLINE, GLOBAL_MAX_DIFF_WHERE

    tt = TTFont(font_path, lazy=False)
    glyf = tt["glyf"]

    glyph_order = tt.getGlyphOrder()
    glyph_to_cps = _build_glyph_to_codepoints(tt)

    for codepoints in glyph_to_cps.values():
        for cp in codepoints:
            GLOBAL_UNICODE_SEEN_COUNTS[cp] += 1

    font_max_diff = -(10**9)

    doc_lines: List[str] = []
    doc_lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    doc_lines.append(f'<Font name="{esc(font_path.name)}">')

    if "fvar" in tt:
        doc_lines.append("  <fvar>")
        fvar_axes_list = tt["fvar"].axes
        for a in fvar_axes_list:
            GLOBAL_FVAR_AXIS_COUNTS[a.axisTag] += 1
            doc_lines.append(
                f'    <axis tag="{esc(a.axisTag)}" min="{a.minValue:g}" default="{a.defaultValue:g}" max="{a.maxValue:g}"/>'
            )
        GLOBAL_FONT_NUM_AXES_COUNTS[len(fvar_axes_list)] += 1
        doc_lines.append("  </fvar>")

    doc_lines.append("  <Glyphs>")

    for gname in glyph_order:
        codepoints = glyph_to_cps.get(gname, [])

        if only_simple_glyphs:
            if not codepoints:
                continue
            if len(codepoints) > 1:
                continue
            if not _glyph_passes_only_simple_glyphs_filter(codepoints):
                continue

        if gname in glyf.glyphs:
            g = glyf[gname]
            if not g.isComposite():
                coords, end_pts, flags = g.getCoordinates(glyf)
                outline_pts = len(coords)
                if outline_pts > 0:
                    var_pts = _max_delta_pt_plus_one(tt, gname)
                    difference_between_variable_points_and_outline_points = (
                        var_pts - outline_pts
                    )

                    if (
                        difference_between_variable_points_and_outline_points
                        > font_max_diff
                    ):
                        font_max_diff = (
                            difference_between_variable_points_and_outline_points
                        )

                    if (
                        difference_between_variable_points_and_outline_points
                        > GLOBAL_MAX_DIFF_PHANTOM_TO_OUTLINE
                    ):
                        GLOBAL_MAX_DIFF_PHANTOM_TO_OUTLINE = (
                            difference_between_variable_points_and_outline_points
                        )
                        GLOBAL_MAX_DIFF_WHERE = (font_path.name, gname)
                        print(
                            f"\033[1m[DETECTED EXTRA PHANTOM POINTS]\033[0m now={GLOBAL_MAX_DIFF_PHANTOM_TO_OUTLINE} "
                            f"(font={GLOBAL_MAX_DIFF_WHERE[0]}, glyph={GLOBAL_MAX_DIFF_WHERE[1]})"
                        )

                        if difference_between_variable_points_and_outline_points > 4:
                            print("should be more than 4 phantom points")
                            exit(1)

        cps_str = _format_codepoints(codepoints) if codepoints else ""
        dec_str = ",".join(str(cp) for cp in codepoints) if codepoints else ""
        chars_str = (
            ",".join(_safe_char_for_xml(cp) for cp in codepoints) if codepoints else ""
        )
        unicode_line = f'      <unicode cps="{esc(cps_str)}" dec="{esc(dec_str)}" chars="{esc(chars_str)}"/>'

        gv_res = gvar_to_ttx_like(
            font_path,
            tt,
            gname,
            indent="        ",
            upm_scale=upm_scale,
            only_axes=only_axes,
            infer_missing_deltas=infer_missing_deltas,
        )
        if gv_res is None:
            gv, axes_used = None, set()
        else:
            gv, axes_used = gv_res

        if only_axes and len(only_axes) > 0:
            if _glyph_has_any_gvar(tt, gname) and gv_res is None:
                continue

        outline = glyph_to_ttx_like(
            tt,
            gname,
            indent="    ",
            skip_composites=skip_composites,
            upm_scale=upm_scale,
            axes_used=axes_used,
        )
        if outline is None:
            continue

        for cp in codepoints:
            GLOBAL_UNICODE_USED_COUNTS[cp] += 1

        doc_lines.append("    <GlyphRecord>")
        doc_lines.append(unicode_line)
        doc_lines.append("      " + outline.replace("\n", "\n      "))

        if gv is not None:
            doc_lines.append("      " + gv.replace("\n", "\n      "))

        doc_lines.append("    </GlyphRecord>")

    doc_lines.append("  </Glyphs>")
    doc_lines.append("</Font>")
    text = "\n".join(doc_lines) + "\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_output:
        with gzip.open(out_path, "wt", encoding="utf-8") as f:
            f.write(text)
    else:
        out_path.write_text(text, encoding="utf-8")

    tt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fonts-dir",
        type=Path,
        required=True,
        help="Folder containing .ttf fonts (variable fonts OK)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output folder (one XML-like file per font)",
    )
    ap.add_argument("--gzip", action="store_true", help="Write .xml.gz instead of .xml")
    ap.add_argument(
        "--skip-composites",
        action="store_true",
        help="Skip composite glyphs (recommended initially)",
    )
    ap.add_argument(
        "--normalize-upm",
        action="store_true",
        help=f"Normalize all coordinates/deltas to UPM={TARGET_UPM}",
    )
    ap.add_argument(
        "--only-simple-glyphs",
        action="store_true",
        help="When set, keep only glyphs whose Unicode is 0-9 / A-Z / a-z (ASCII alnum)",
    )
    ap.add_argument(
        "--only-axes",
        type=str,
        default="",
        help='Comma-separated list of axes to keep in gvar tuples (e.g. "wdth,slnt"). '
        "When set, any tuple containing an axis outside this set is dropped.",
    )
    ap.add_argument(
        "--dont-infer-missing-deltas",
        action="store_false",
        dest="infer_missing_deltas",
        default=True,
        help=(
            "Disable inferring missing deltas. By default, deltas for ALL points in each gvar tuple are "
            "inferred (simple glyphs only, per spec). This fills missing outline-point deltas using the "
            "gvar inferred-deltas algorithm and emits 0,0 for missing phantom-point deltas."
        ),
    )
    args = ap.parse_args()

    only_axes_set: Set[str] = {
        a.strip() for a in args.only_axes.split(",") if a.strip()
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)

    font_paths = [
        p for p in args.fonts_dir.rglob("*.ttf") if "axisregistry/tests" not in str(p)
    ]

    print(f"Found {len(font_paths)} .ttf files")

    seen_names = set()
    for fp in font_paths:
        if fp.name in seen_names:
            print(f"[ERROR] Duplicate font name found: {fp.name}")
            exit(1)
        seen_names.add(fp.name)

    total_fonts = 0
    used_fonts = 0

    for idx, fp in enumerate(font_paths):
        suffix = ".xml.gz" if args.gzip else ".xml"

        if False:
            if not "PlusJakartaSans[wght]" in fp.name:
                continue

        total_fonts += 1

        if False:
            if total_fonts >= 5:
                break

        try:
            tt = TTFont(fp, lazy=True)
            upm = tt["head"].unitsPerEm
            tt.close()
        except Exception as e:
            print(f"[skip] {fp}: cannot read head table ({e})")
            continue

        if args.normalize_upm:
            if upm <= 0:
                print(
                    f"\033[1m[WARNING]\033[0m {fp.name}: unitsPerEm={upm} is invalid — skipping font"
                )
                exit(0)
            upm_scale = TARGET_UPM / float(upm)
        else:
            upm_scale = 1.0
            if upm != TARGET_UPM:
                print(
                    f"\033[1m[WARNING]\033[0m {fp.name}: unitsPerEm={upm}, expected {TARGET_UPM} — "
                    f"skipping font (use --normalize-upm to keep it)"
                )
                continue

        out_path = args.out_dir / (fp.stem + suffix)
        try:
            extract_font_ttx_like(
                fp,
                out_path,
                gzip_output=args.gzip,
                skip_composites=args.skip_composites,
                upm_scale=upm_scale,
                only_simple_glyphs=args.only_simple_glyphs,
                only_axes=only_axes_set,
                infer_missing_deltas=args.infer_missing_deltas,
            )
            print(f"[ok] {idx}: {fp.name} -> {out_path.name} [orig upm={upm}]")
            used_fonts += 1
        except Exception as e:
            print(f"[skip] {fp}: {e}")

    print(
        f"\nSummary:\n"
        f"  Fonts scanned: {total_fonts}\n"
        f"  Fonts used   : {used_fonts}\n"
        f"  Fonts skipped: {total_fonts - used_fonts}\n"
    )

    print(
        f"\nGLOBAL max diff across all fonts/glyphs: {GLOBAL_MAX_DIFF_PHANTOM_TO_OUTLINE} at font={GLOBAL_MAX_DIFF_WHERE[0]} glyph={GLOBAL_MAX_DIFF_WHERE[1]}"
    )

    print("\nAxis <coord> seen in dataset (ALL tuples, before --only-axes filter):")
    if GLOBAL_AXIS_COORD_SEEN_COUNTS:
        total_seen = sum(GLOBAL_AXIS_COORD_SEEN_COUNTS.values())
        for axis, cnt in GLOBAL_AXIS_COORD_SEEN_COUNTS.most_common():
            pct = 100.0 * cnt / total_seen
            print(f"  {axis:4s} : {cnt:8d}  ({pct:6.2f}%)")
        print(f"  TOTAL: {total_seen}")
    else:
        print("  (none)")

    print("\nAxis <coord> usage (after --only-axes filter):")
    if GLOBAL_AXIS_COORD_COUNTS:
        total = sum(GLOBAL_AXIS_COORD_COUNTS.values())
        for axis, cnt in GLOBAL_AXIS_COORD_COUNTS.most_common():
            pct = 100.0 * cnt / total
            print(f"  {axis:4s} : {cnt:8d}  ({pct:6.2f}%)")
        print(f"  TOTAL: {total}")
    else:
        print("  (none)")

    print("\nAxes present in fvar (number of fonts containing the axis):")
    if GLOBAL_FVAR_AXIS_COUNTS:
        for axis, cnt in GLOBAL_FVAR_AXIS_COUNTS.most_common():
            print(f"  {axis:4s} : {cnt}")
    else:
        print("  (none)")

    print("\nFont axis-count distribution (how many fvar axes each font has):")
    if GLOBAL_FONT_NUM_AXES_COUNTS:
        total_fonts_with_fvar = sum(GLOBAL_FONT_NUM_AXES_COUNTS.values())
        for n_axes in sorted(GLOBAL_FONT_NUM_AXES_COUNTS.keys()):
            cnt = GLOBAL_FONT_NUM_AXES_COUNTS[n_axes]
            pct = 100.0 * cnt / total_fonts_with_fvar
            print(f"  {n_axes} axis/axes : {cnt:6d}  ({pct:6.2f}%)")
        print(f"  TOTAL fonts: {total_fonts_with_fvar}")
    else:
        print("  (none)")

    print(
        "\nTuple axis-count distribution (how many axes per tuple), for ALL axes in data:"
    )
    if GLOBAL_TUPLE_NUM_AXES_COUNTS:
        total_tuples = sum(GLOBAL_TUPLE_NUM_AXES_COUNTS.values())
        for n_axes in sorted(GLOBAL_TUPLE_NUM_AXES_COUNTS.keys()):
            cnt = GLOBAL_TUPLE_NUM_AXES_COUNTS[n_axes]
            pct = 100.0 * cnt / total_tuples
            print(f"  {n_axes} axis/axes : {cnt:8d}  ({pct:6.2f}%)")
        print(f"  TOTAL tuples: {total_tuples}")
        unusual_pct = 100.0 * GLOBAL_TUPLES_WITH_UNUSUAL_PEAK / total_tuples
        print(
            f"  Tuples with at least one unusual peak value "
            f"(not -1/0/+1): {GLOBAL_TUPLES_WITH_UNUSUAL_PEAK:8d}  ({unusual_pct:6.2f}%)"
        )
        default_tent_pct = 100.0 * GLOBAL_TUPLES_DEFAULT_TENT / total_tuples
        explicit_tent_pct = 100.0 * GLOBAL_TUPLES_WITH_EXPLICIT_TENT / total_tuples
        explicit_min_pct = 100.0 * GLOBAL_TUPLES_WITH_EXPLICIT_MIN / total_tuples
        explicit_max_pct = 100.0 * GLOBAL_TUPLES_WITH_EXPLICIT_MAX / total_tuples
        explicit_both_pct = 100.0 * GLOBAL_TUPLES_WITH_EXPLICIT_BOTH / total_tuples
        print(
            f"  Tuples with default tent (peak-only coords): "
            f"{GLOBAL_TUPLES_DEFAULT_TENT:8d}  ({default_tent_pct:6.2f}%)"
        )
        print(
            f"  Tuples with explicit tent override (min and/or max): "
            f"{GLOBAL_TUPLES_WITH_EXPLICIT_TENT:8d}  ({explicit_tent_pct:6.2f}%)"
        )
        print(
            f"  Tuples overriding left boundary (min): "
            f"{GLOBAL_TUPLES_WITH_EXPLICIT_MIN:8d}  ({explicit_min_pct:6.2f}%)"
        )
        print(
            f"  Tuples overriding right boundary (max): "
            f"{GLOBAL_TUPLES_WITH_EXPLICIT_MAX:8d}  ({explicit_max_pct:6.2f}%)"
        )
        print(
            f"  Tuples overriding both boundaries (min+max): "
            f"{GLOBAL_TUPLES_WITH_EXPLICIT_BOTH:8d}  ({explicit_both_pct:6.2f}%)"
        )
    else:
        print("  (none)")

    distinct_unicodes_seen = len(GLOBAL_UNICODE_SEEN_COUNTS)
    distinct_unicodes_used = len(GLOBAL_UNICODE_USED_COUNTS)
    total_glyphs_seen = sum(GLOBAL_UNICODE_SEEN_COUNTS.values())
    total_glyphs_used = sum(GLOBAL_UNICODE_USED_COUNTS.values())
    print(f"\nUnicode Statistics:")
    print(
        f"  Distinct unicodes seen: {distinct_unicodes_seen}, total glyphs seen: {total_glyphs_seen}"
    )
    print(
        f"  Distinct unicodes used: {distinct_unicodes_used}, total glyphs used: {total_glyphs_used}"
    )

    seen_csv_path = "tmp_unicodes_seen.csv"
    used_csv_path = "tmp_unicodes_used.csv"

    with open(seen_csv_path, "w", encoding="utf-8") as f:
        f.write("unicode_id,count\n")
        for cp, cnt in GLOBAL_UNICODE_SEEN_COUNTS.most_common():
            f.write(f"{cp},{cnt}\n")
    print(f"\nSaved seen unicodes to {seen_csv_path}")

    with open(used_csv_path, "w", encoding="utf-8") as f:
        f.write("unicode_id,count\n")
        for cp, cnt in GLOBAL_UNICODE_USED_COUNTS.most_common():
            f.write(f"{cp},{cnt}\n")
    print(f"Saved used unicodes to {used_csv_path}")


if __name__ == "__main__":
    main()
