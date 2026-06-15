from __future__ import annotations


import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
from itertools import product, combinations

from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables._g_v_a_r import TupleVariation
from fontTools.ttLib.tables._f_v_a_r import Axis

try:
    from fontTools.otlLib.builder import buildStatTable
except Exception:
    buildStatTable = None

import torch
from niv_model import GlyphDeltaRegressor
from niv_model_train import compute_tuple_weight


def _parse_axis_list(s: str) -> List[str]:
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def _parse_unicode_range(s: str) -> Union[Tuple[int, int], Set[int]]:

    s = s.strip().upper().replace("U+", "0X")

    def _one(x: str) -> int:
        x = x.strip()
        if x.startswith("0X"):
            return int(x, 16)
        if any(c in "ABCDEF" for c in x):
            return int(x, 16)
        return int(x, 10)

    if "," in s:
        try:
            values: Set[int] = set()
            for tok in (t.strip() for t in s.split(",") if t.strip()):
                if "-" in tok:
                    m = re.match(r"^\s*([0-9A-FX]+)\s*-\s*([0-9A-FX]+)\s*$", tok)
                    if not m:
                        raise ValueError(f"Bad range token: {tok!r}")
                    a, b = m.group(1), m.group(2)
                    lo, hi = _one(a), _one(b)
                    if lo > hi:
                        lo, hi = hi, lo
                    values.update(range(lo, hi + 1))
                else:
                    values.add(_one(tok))
            return values
        except ValueError as e:
            raise ValueError(f"Bad --unicode-range comma list: {s!r}. Error: {e}")

    m = re.match(r"^\s*([0-9A-FX]+)\s*-\s*([0-9A-FX]+)\s*$", s)
    if not m:
        raise ValueError(
            f"Bad --unicode-range: {s!r}. Expected like 32-126, 0x20-0x7E, or comma list with singles/ranges like 65,66-68,90."
        )
    a, b = m.group(1), m.group(2)

    lo, hi = _one(a), _one(b)
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _resolve_checkpoint_paths(model_arg: str) -> Tuple[Path, Path]:
    p = Path(model_arg)
    if p.is_dir():
        model_pt = p / "model.pt"
        cfg = p / "config.json"
    else:
        model_pt = p
        cfg = p.parent / "config.json"
    if not model_pt.exists():
        raise FileNotFoundError(f"--model not found: {model_pt}")
    if not cfg.exists():
        raise FileNotFoundError(f"config.json not found next to checkpoint: {cfg}")
    return model_pt, cfg


def add_fvar(tt: TTFont, axis_tags: List[str]) -> None:
    if "fvar" in tt:
        return

    fvar = newTable("fvar")
    fvar.axes = []
    fvar.instances = []

    name_table = tt["name"]

    def _add_name(s: str) -> int:

        for rec in name_table.names:
            try:
                if rec.toUnicode() == s:
                    return int(rec.nameID)
            except Exception:
                pass
        used = {int(r.nameID) for r in name_table.names}
        name_id = 256
        while name_id in used:
            name_id += 1
        name_table.setName(s, name_id, 3, 1, 0x409)
        name_table.setName(s, name_id, 1, 0, 0)
        return int(name_id)

    for i, tag in enumerate(axis_tags):
        ax = Axis()
        ax.axisTag = tag
        ax.minValue = -1.0
        ax.defaultValue = 0.0
        ax.maxValue = 1.0
        ax.flags = 0
        ax.axisNameID = _add_name(tag)
        fvar.axes.append(ax)

    tt["fvar"] = fvar


def add_avar_identity(tt: TTFont, axis_tags: List[str]) -> None:

    if not axis_tags:
        return
    if "avar" in tt:
        return

    avar = newTable("avar")

    avar.version = 0x00010000

    avar.segments = {tag: {-1.0: -1.0, 0.0: 0.0, 1.0: 1.0} for tag in axis_tags}

    tt["avar"] = avar


def add_gvar_empty(tt: TTFont) -> None:
    if "gvar" in tt:
        return
    gvar = newTable("gvar")
    gvar.version = 1
    gvar.variations = {}
    tt["gvar"] = gvar


def add_STAT_minimal(tt: TTFont, axis_tags: List[str]) -> None:
    if "STAT" in tt:
        return
    if buildStatTable is None:
        return
    axes = [{"tag": t, "name": t, "ordering": i} for i, t in enumerate(axis_tags)]
    try:
        buildStatTable(tt, axes)
    except TypeError:
        try:
            buildStatTable(tt, axes=axes)
        except Exception:
            return
    except Exception:
        return


@dataclass
class ModelCfg:
    d_model: int
    num_layers: int
    num_heads: int
    ff_mult: int
    dropout: float
    cond_method: str
    pos_enc: str
    point_mlp_hidden: int
    cond_mlp_hidden: int
    max_points: int
    axes: List[str]
    point_contour_info: int
    axisembed_method: str
    prediction_type: str
    split_method: str
    combinatorial_max_order: int
    polar_regression: bool = False
    lineloss: Optional[float] = None
    coordnorm: bool = False


def _infer_from_state_dict(sd: Dict[str, "torch.Tensor"]) -> Tuple[int, int]:
    point_feat_dim = None
    for k, v in sd.items():
        if (
            k.endswith("point_mlp.0.weight")
            and hasattr(v, "shape")
            and len(v.shape) == 2
        ):
            point_feat_dim = int(v.shape[1])
            break
    if point_feat_dim is None:

        cands = []
        for k, v in sd.items():
            if hasattr(v, "shape") and len(v.shape) == 2:
                in_f = int(v.shape[1])
                if 2 <= in_f <= 16:
                    cands.append(in_f)
        if not cands:
            raise RuntimeError("Could not infer point_feat_dim from checkpoint.")
        point_feat_dim = min(cands)

    num_contour_id_plus_three = None
    for k, v in sd.items():
        if (
            "contour" in k.lower()
            and "embed" in k.lower()
            and k.endswith("weight")
            and hasattr(v, "shape")
            and len(v.shape) == 2
        ):
            num_contour_id_plus_three = int(v.shape[0])
            break
    if num_contour_id_plus_three is None:
        num_contour_id_plus_three = 0
    return int(point_feat_dim), int(num_contour_id_plus_three)


def _load_model(model_arg: str):
    if torch is None or GlyphDeltaRegressor is None:
        raise RuntimeError(
            "PyTorch and/or GlyphDeltaRegressor not available in this environment."
        )

    model_pt, cfg_path = _resolve_checkpoint_paths(model_arg)
    cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    if "lineloss" not in cfg_raw and "loss1" in cfg_raw:
        cfg_raw["lineloss"] = cfg_raw["loss1"]
    cfg_raw.pop("loss1", None)
    if "combinatorial_max_order" not in cfg_raw:
        raise KeyError(
            f"Missing required key 'combinatorial_max_order' in config.json: {cfg_path}"
        )
    cfg = ModelCfg(**cfg_raw)
    print(f"[model] loaded config: {json.dumps(cfg_raw)}")

    sd = torch.load(model_pt, map_location="cpu")
    if not isinstance(sd, dict):
        raise RuntimeError(f"Unexpected checkpoint type: {type(sd)}")

    point_feat_dim, num_contour_id_plus_three = _infer_from_state_dict(sd)

    model = GlyphDeltaRegressor(
        num_axes=len(cfg.axes),
        max_points=int(cfg.max_points),
        d_model=int(cfg.d_model),
        num_layers=int(cfg.num_layers),
        num_heads=int(cfg.num_heads),
        ff_mult=int(cfg.ff_mult),
        dropout=float(cfg.dropout),
        cond_method=str(cfg.cond_method),
        axisembed_method=str(cfg.axisembed_method),
        pos_enc=str(cfg.pos_enc),
        point_mlp_hidden=int(cfg.point_mlp_hidden),
        cond_mlp_hidden=int(cfg.cond_mlp_hidden),
        point_contour_info=int(cfg.point_contour_info),
        num_contour_id_plus_three=int(num_contour_id_plus_three),
        point_feat_dim=int(point_feat_dim),
        polar_regression=bool(cfg.polar_regression),
        coordnorm=bool(cfg.coordnorm),
    )
    if True:

        if any(k.startswith("encoder.") for k in sd.keys()) and not any(
            k.startswith("cond_transformer_encoder.") for k in sd.keys()
        ):
            sd = {
                (
                    k.replace("encoder.", "cond_transformer_encoder.", 1)
                    if k.startswith("encoder.")
                    else k
                ): v
                for k, v in sd.items()
            }
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, cfg, point_feat_dim, num_contour_id_plus_three


def _build_contour_id_from_endpts(end_pts: List[int]) -> List[int]:
    if not end_pts:
        return []
    contour_id: List[int] = []
    start = 0
    for ci, end in enumerate(end_pts):
        L = end - start + 1
        contour_id.extend([ci] * max(0, L))
        start = end + 1
    return contour_id


def _compute_pos_in_contour(contour_id: List[int]) -> List[float]:
    n = len(contour_id)
    pos = [0.0] * n
    i = 0
    while i < n:
        cid = contour_id[i]
        j = i + 1
        while j < n and contour_id[j] == cid:
            j += 1
        L = j - i
        denom = float(max(1, L - 1))
        for k in range(L):
            pos[i + k] = float(k) / denom
        i = j
    return pos


@torch.no_grad()
def _predict_deltas_for_axes(
    model,
    cfg: ModelCfg,
    *,
    points_xy: List[Tuple[int, int]],
    oncurve: List[int],
    contour_id: List[int],
    axis_ids: List[int],
    axis_values: List[float],
    num_contours_base: int,
    gname_for_debugging: Optional[str] = None,
    units_per_em: int,
) -> Tuple[List[Tuple[int, int]], float]:

    print(
        f"[model] predicting deltas, name={gname_for_debugging}, for axis_ids={axis_ids} values={axis_values} with {len(points_xy)} points and {num_contours_base} contours"
    )

    import torch as T

    device = next(model.parameters()).device
    N_base = len(points_xy)
    N = N_base + 4

    upm = float(max(1, int(units_per_em)))
    coord_scale = 2.0 / upm

    xy = T.tensor(points_xy, dtype=T.float32)
    oc = T.tensor(oncurve, dtype=T.float32).view(N_base, 1)
    xy_norm = xy * coord_scale - 1.0
    pts_base_3 = T.cat([xy_norm, oc], dim=-1)

    phantom_xy = T.zeros((4, 2), dtype=T.float32)
    phantom_oc = T.full((4, 1), -1.0, dtype=T.float32)
    pts_ph_3 = T.cat([phantom_xy, phantom_oc], dim=-1)

    pts3 = T.cat([pts_base_3, pts_ph_3], dim=0)

    cids_base = T.tensor(contour_id, dtype=T.long)
    phantom_tok_id = num_contours_base

    tok = T.empty((N,), dtype=T.long)
    if num_contours_base > 0 and N_base > 0:
        tok[:N_base] = T.clamp(cids_base, min=0, max=num_contours_base - 1)
    else:
        tok[:N_base] = 0
    tok[N_base:] = phantom_tok_id

    if int(cfg.point_contour_info) in (2, 3):
        extra = T.zeros((N, 1), dtype=T.float32)
        if N_base > 0:
            if int(cfg.point_contour_info) == 2:
                extra[0, 0] = 1.0
                if N_base > 1:
                    same_prev = cids_base[1:] == cids_base[:-1]
                    extra[1:N_base, 0] = (~same_prev).float()
                extra[N_base:, 0] = 0.0
            else:
                pos = T.tensor(_compute_pos_in_contour(contour_id), dtype=T.float32)
                extra[:N_base, 0] = pos
                extra[N_base:, 0] = 0.0
        pts = T.cat([pts3, extra], dim=-1)
    else:
        pts = pts3

    points = pts.unsqueeze(0).to(device)
    contour_tok_id = tok.unsqueeze(0).to(device)

    num_axes = len(axis_ids)
    axis_id_t = T.tensor([axis_ids], dtype=T.long, device=device)
    axis_val_t = T.tensor([axis_values], dtype=T.float32, device=device)
    pad_mask = T.zeros((1, N), dtype=T.bool, device=device)
    axis_mask = T.ones((1, num_axes), dtype=T.bool, device=device)

    forward_t0 = time.perf_counter()
    pred = model(
        points,
        contour_tok_id,
        axis_id_t,
        axis_val_t,
        pad_mask,
        axis_mask,
        combinatorial_max_order=cfg.combinatorial_max_order,
    )
    forward_seconds = time.perf_counter() - forward_t0

    pred = pred[0].to("cpu")
    pred_upm = pred * (upm / 2.0)

    out: List[Tuple[int, int]] = []
    for j in range(N):
        out.append(
            (
                int(round(float(pred_upm[j, 0].item()))),
                int(round(float(pred_upm[j, 1].item()))),
            )
        )
    return out, forward_seconds


def inject_model_glyph_variations(
    tt: TTFont,
    *,
    model_arg: Optional[str],
    models_args: Optional[List[str]],
    args_requested_axes: List[str],
    unicode_lo: Optional[int],
    unicode_hi: Optional[int],
    unicode_set: Optional[Set[int]],
    delta_level: int,
    prediction_type: str,
) -> None:
    if (model_arg is None) == (models_args is None):
        raise ValueError("Provide exactly one of --model or --models")

    font_upm = int(tt["head"].unitsPerEm)

    per_axis_models: List[Tuple[object, ModelCfg, int, int]] = []

    if model_arg is not None:
        model, cfg, _, num_contour_id_plus_three = _load_model(model_arg)
        axis_to_id = {a: i for i, a in enumerate(cfg.axes)}
        for t in args_requested_axes:
            if t not in axis_to_id:
                raise ValueError(
                    f"Axis {t!r} requested but model config axes are {cfg.axes}"
                )
            per_axis_models.append(
                (model, cfg, max(0, int(num_contour_id_plus_three) - 2), axis_to_id[t])
            )
    else:
        print("Warning: multiple models is are used.")
        if not models_args:
            raise ValueError("--models provided but empty")
        if len(models_args) != len(args_requested_axes):
            raise ValueError("Number of models in --models must match number of axes")

        for axis_tag, model_path in zip(args_requested_axes, models_args):
            model, cfg, _, num_contour_id_plus_three = _load_model(model_path)
            axis_to_id = {a: i for i, a in enumerate(cfg.axes)}
            if axis_tag not in axis_to_id:
                raise ValueError(
                    f"Axis {axis_tag!r} requested but model config axes are {cfg.axes} for model {model_path}"
                )
            per_axis_models.append(
                (
                    model,
                    cfg,
                    max(0, int(num_contour_id_plus_three) - 2),
                    axis_to_id[axis_tag],
                )
            )

    axis_meta = [
        (axis_tag, per_axis_models[i][3])
        for i, axis_tag in enumerate(args_requested_axes)
    ]
    raw_permutations = []
    for values in product((-1.0, 0.0, +1.0), repeat=len(axis_meta)):
        entries = [
            {
                "axis": axis_meta[i][0],
                "axis_id": axis_meta[i][1],
                "value": float(values[i]),
            }
            for i in range(len(axis_meta))
        ]
        nonzero_indices = [i for i, v in enumerate(values) if v != 0.0]
        raw_permutations.append(
            {
                "values": [float(v) for v in values],
                "entries": entries,
                "nonzero_indices": nonzero_indices,
                "nonzero_count": len(nonzero_indices),
            }
        )

    raw_permutations.sort(key=lambda p: p["nonzero_count"])
    axis_value_permutations = [
        p for p in raw_permutations if 0 < p["nonzero_count"] <= delta_level
    ]

    print(
        f"[model] prepared {len(axis_value_permutations)} permutations (delta_level={delta_level}) "
        f"from {len(raw_permutations)} total (removed all-zero and >delta_level)."
    )
    for i, perm in enumerate(axis_value_permutations):
        desc = ", ".join(
            f"{e['axis']} (id={e['axis_id']}): {e['value']:+.1f}"
            for e in perm["entries"]
        )
        print(f"[combo {i:03d}] nonzeros={perm['nonzero_count']} values: {desc}")

    best_cmap = tt.getBestCmap() or {}
    cmap = {int(cp): g for cp, g in best_cmap.items()}

    win_symbol = None
    mac_roman = None
    try:
        for t in tt["cmap"].tables:
            if t.platformID == 3 and t.platEncID == 0:
                win_symbol = t.cmap
            elif t.platformID == 1 and t.platEncID == 0:
                mac_roman = t.cmap
    except Exception:
        win_symbol = None
        mac_roman = None

    if win_symbol:
        for cp, g in win_symbol.items():
            cp_i = int(cp)
            cmap.setdefault(cp_i, g)
            if 0 <= cp_i <= 0xFF:
                cmap.setdefault(cp_i + 0xF000, g)
            elif 0xF000 <= cp_i <= 0xF0FF:
                cmap.setdefault(cp_i - 0xF000, g)

    if mac_roman:
        for cp, g in mac_roman.items():
            cp_i = int(cp)
            if 0 <= cp_i <= 0xFF:
                cmap.setdefault(cp_i, g)

    if unicode_set is not None:
        items = [(cp, g) for cp, g in cmap.items() if int(cp) in unicode_set]
        items.sort(key=lambda x: x[0])
        if not items:
            print(
                f"[model] No cmap entries for specified unicode values: {sorted(unicode_set)}"
            )
            return
    else:
        items = [
            (cp, g) for cp, g in cmap.items() if unicode_lo <= int(cp) <= unicode_hi
        ]
        items.sort(key=lambda x: x[0])
        if not items:
            print(f"[model] No cmap entries in U+{unicode_lo:04X}-U+{unicode_hi:04X}")
            return

    glyf = tt["glyf"]
    gvar = tt["gvar"]

    n_added = 0
    n_skipped_comp = 0
    total_forward_seconds = 0.0

    for cp, glyph_name in items:
        if glyph_name not in glyf.glyphs:
            continue
        g = glyf[glyph_name]
        if g.isComposite():
            n_skipped_comp += 1
            continue

        coords, end_pts, flags = g.getCoordinates(glyf)
        end_pts = list(end_pts) if end_pts is not None else []
        if not end_pts:
            continue

        N_base = int(end_pts[-1]) + 1
        pts_xy = [(int(coords[i][0]), int(coords[i][1])) for i in range(N_base)]
        oncurve = [1 if (int(flags[i]) & 0x01) else 0 for i in range(N_base)]
        contour_id = _build_contour_id_from_endpts(end_pts)
        if len(contour_id) != N_base:
            continue

        existingGvarVariations = gvar.variations.get(glyph_name)
        if existingGvarVariations is None:
            existingGvarVariations = []
            gvar.variations[glyph_name] = existingGvarVariations

        model, cfg, num_contours_base, _ = per_axis_models[0]

        for perm in axis_value_permutations:
            axes_indices = perm["nonzero_indices"]

            input_pts_xy = pts_xy
            input_oncurve = oncurve
            deltas_to_subtract = None

            if prediction_type in ("residual", "total") and existingGvarVariations:
                axis_values_dict = {
                    args_requested_axes[idx]: perm["values"][idx]
                    for idx in axes_indices
                }

                deltas_to_apply = [(0, 0) for _ in range(len(pts_xy) + 4)]

                for otherTupleVariation in existingGvarVariations:

                    other_tuple_coords = {}
                    for axis_name, (
                        min_val,
                        peak_val,
                        max_val,
                    ) in otherTupleVariation.axes.items():
                        other_tuple_coords[axis_name] = {
                            "min": min_val,
                            "value": peak_val,
                            "max": max_val,
                        }
                    weight = compute_tuple_weight(other_tuple_coords, axis_values_dict)
                    if weight >= 0:
                        for pt_idx, (dx, dy) in enumerate(
                            otherTupleVariation.coordinates
                        ):
                            curr_dx, curr_dy = deltas_to_apply[pt_idx]
                            deltas_to_apply[pt_idx] = (
                                curr_dx + weight * dx,
                                curr_dy + weight * dy,
                            )
                    else:
                        raise RuntimeError(
                            f"Negative weight encountered in {prediction_type} mode"
                        )

                if prediction_type == "residual":

                    input_pts_xy = [
                        (
                            int(pts_xy[pt_idx][0]) + int(deltas_to_apply[pt_idx][0]),
                            int(pts_xy[pt_idx][1]) + int(deltas_to_apply[pt_idx][1]),
                        )
                        for pt_idx in range(len(pts_xy))
                    ]
                elif prediction_type == "total":

                    deltas_to_subtract = deltas_to_apply

            axis_ids_ordered = [per_axis_models[idx][3] for idx in axes_indices]
            axis_vals_ordered = [float(perm["values"][idx]) for idx in axes_indices]

            deltas, forward_seconds = _predict_deltas_for_axes(
                model,
                cfg,
                points_xy=input_pts_xy,
                oncurve=input_oncurve,
                contour_id=contour_id,
                axis_ids=axis_ids_ordered,
                axis_values=axis_vals_ordered,
                num_contours_base=num_contours_base,
                gname_for_debugging=glyph_name,
                units_per_em=font_upm,
            )
            total_forward_seconds += forward_seconds

            if deltas_to_subtract is not None:
                deltas = [
                    (
                        int(round(dx - deltas_to_subtract[pt_idx][0])),
                        int(round(dy - deltas_to_subtract[pt_idx][1])),
                    )
                    for pt_idx, (dx, dy) in enumerate(deltas)
                ]

            coords = {}
            for axis_idx in axes_indices:
                peak = perm["values"][axis_idx]
                if peak > 0:
                    coords[args_requested_axes[axis_idx]] = (
                        0,
                        float(peak),
                        float(peak),
                    )
                else:
                    coords[args_requested_axes[axis_idx]] = (
                        float(peak),
                        float(peak),
                        0,
                    )

            otherTupleVariation = TupleVariation(coords, deltas)
            existingGvarVariations.append(otherTupleVariation)
            n_added += 1

    print(f"[model] injected tuples: {n_added} (skipped composites: {n_skipped_comp})")
    avg_forward_seconds = (total_forward_seconds / n_added) if n_added > 0 else 0.0
    print(
        f"[model] average forward time per tuple: {avg_forward_seconds:.6f} s "
        f"(total forward time: {total_forward_seconds:.6f} s over {n_added} tuples)"
    )


def make_variable_skeleton(
    font_path: Path,
    args_requested_axes: List[str],
    model: Optional[str],
    models: Optional[List[str]],
    unicode_range: Optional[str],
    delta_level: int,
    prediction_type: str,
) -> Path:
    tt = TTFont(str(font_path))

    if args_requested_axes:
        add_fvar(tt, args_requested_axes)
        add_avar_identity(tt, args_requested_axes)
    else:
        raise NotImplementedError("At least one axis must be specified with --axes")

    add_gvar_empty(tt)
    add_STAT_minimal(tt, args_requested_axes)

    if model is not None and models is not None:
        raise ValueError("Provide only one of --model or --models")

    if model is not None or models:
        if unicode_range is None:
            raise ValueError("--model/--models requires --unicode-range")
        parsed = _parse_unicode_range(unicode_range)

        if isinstance(parsed, set):
            lo, hi, unicode_set = None, None, parsed
        else:
            lo, hi, unicode_set = parsed[0], parsed[1], None

        inject_model_glyph_variations(
            tt,
            model_arg=model,
            models_args=models,
            args_requested_axes=args_requested_axes,
            unicode_lo=lo,
            unicode_hi=hi,
            unicode_set=unicode_set,
            delta_level=delta_level,
            prediction_type=prediction_type,
        )

    out_path = Path(font_path.with_suffix("").as_posix() + "_var.ttf")
    tt.save(str(out_path))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("font_path", type=Path, help="Path to a .ttf")
    ap.add_argument(
        "--axes",
        type=str,
        default="",
        help="Comma-separated axis tags, e.g. wght,wdth,slnt,opsz",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="Checkpoint dir or model.pt from train_glyph_variations_transformer.py",
    )
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list ofe checkpoints, one per axis, matching --axes order",
    )
    ap.add_argument(
        "--unicode-range",
        type=str,
        default=None,
        help="Unicode selection: single range (e.g. 32-126 or 0x20-0x7E) or comma-separated singles/ranges (e.g. 65,66-68,0x131)",
    )
    ap.add_argument(
        "--delta-level",
        type=int,
        default=1,
        help="1=single-axis deltas only, 2=add two-axis deltas, etc.(default: 1)",
    )
    ap.add_argument(
        "--prediction-type",
        type=str,
        default="total",
        choices=["tuple", "residual", "total"],
        help="Prediction type: 'tuple' (default) = model predicts single tuple delta; "
        "'residual' = for two-axis tuples, accumulate single-axis deltas and use as input; "
        "'total' = accumulate single-axis deltas and subtract from prediction",
    )
    args = ap.parse_args()
    args_requested_axes = _parse_axis_list(args.axes)
    model_list = _parse_axis_list(args.models) if args.models else []

    if args.model and model_list:
        raise ValueError("--model and --models are mutually exclusive")
    if model_list and len(model_list) != len(args_requested_axes):
        raise ValueError(
            "Number of models in --models must match number of axes in --axes"
        )

    out_path = make_variable_skeleton(
        args.font_path,
        args_requested_axes,
        args.model,
        model_list if model_list else None,
        args.unicode_range,
        args.delta_level,
        args.prediction_type,
    )
    print(f"[ok] wrote: {out_path}")


if __name__ == "__main__":
    main()
