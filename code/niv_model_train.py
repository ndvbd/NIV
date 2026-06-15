from __future__ import annotations

import argparse
from curses import meta
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set


import os
import pickle
import re
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from niv_model import (
    GlyphDeltaRegressor,
    CondTransformerEncoder,
    CondTransformerEncoderLayer,
    SinusoidalPosEnc,
    AxisIdAndValueEmbedder,
    huber_loss,
)


def compute_loss(
    pred: torch.Tensor, target: torch.Tensor, loss_name: str, huber_delta: float
) -> torch.Tensor:

    if loss_name == "huber":
        return huber_loss(pred, target, delta=huber_delta).sum(dim=-1)
    elif loss_name == "mse":
        return F.mse_loss(pred, target, reduction="none").sum(dim=-1)
    else:
        raise ValueError(f"Unknown loss: {loss_name}. Choose 'huber' or 'mse'.")


from niv_line_loss import (
    enrich_with_line_constraints,
    compute_line_preservation_loss,
    apply_line_preservation_inference,
)


def maybe_init_wandb(args):

    if not args.wandb:
        print("[wandb] not enabled")
        return None

    print("[wabnd] initializing wandb")
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )

    wandb.define_metric("eval/loss", summary="min")
    return run


def maybe_wandb_log(args, data: dict, *, step: int) -> None:

    if not args.wandb:
        return
    try:
        import wandb

        wandb.log(data, step=step)
    except Exception:
        pass


SEED = 42


SOURCE_UPM = 2048


from shared import (
    ParsedGlyph,
    ParsedTuple,
    iter_all_pairs_with_glyph_name,
    iter_all_pairs,
    parse_ttglyph_inner,
    parse_gvar_inner,
)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_cuda() -> bool:
    return torch.cuda.is_available()


def compute_axis_weight(
    coord_dict: Dict[str, float], axis_name: str, axis_value: float
) -> float:

    if "min" in coord_dict and "max" in coord_dict:

        min_val = float(coord_dict["min"])
        peak_val = float(coord_dict["value"])
        max_val = float(coord_dict["max"])
    else:

        peak_val = float(coord_dict["value"])
        if peak_val >= 0:
            min_val = 0.0
            max_val = 1.0
        else:
            min_val = -1.0
            max_val = 0.0

    if axis_value <= min_val or axis_value >= max_val:
        if abs(axis_value - peak_val) < 1e-9:
            return 1.0
        return 0.0

    if abs(axis_value - peak_val) < 1e-9:
        return 1.0

    if axis_value < peak_val:

        if abs(peak_val - min_val) < 1e-9:
            return 0.0
        return (axis_value - min_val) / (peak_val - min_val)
    else:

        if abs(max_val - peak_val) < 1e-9:
            return 0.0
        return (max_val - axis_value) / (max_val - peak_val)


def compute_tuple_weight(
    tuple_coords: Dict[str, float], axis_values: Dict[str, float]
) -> float:

    weight = 1.0
    for axis_name, coord_dict in tuple_coords.items():
        axis_val = axis_values.get(axis_name, 0.0)
        axis_weight = compute_axis_weight(coord_dict, axis_name, axis_val)
        weight *= axis_weight
        if weight == 0.0:
            break
    return weight


@dataclass
class AxisSpec:
    name_to_id: Dict[str, int]
    id_to_name: List[str]

    @classmethod
    def from_axis_list(cls, axes: List[str]) -> "AxisSpec":
        axes = sorted(set(axes))
        return cls(name_to_id={a: i for i, a in enumerate(axes)}, id_to_name=axes)


@dataclass
class RegressionBatch:
    points: torch.Tensor
    contour_id: torch.Tensor
    contour_tok_id: torch.Tensor
    axis_id: torch.Tensor
    axis_value: torch.Tensor
    axis_mask: torch.Tensor
    deltas: torch.Tensor
    pad_mask: torch.Tensor
    lengths: torch.Tensor
    meta: List[Dict]


def move_batch_to_device(b: RegressionBatch, device: torch.device) -> RegressionBatch:
    return RegressionBatch(
        points=b.points.to(device, non_blocking=True),
        contour_id=b.contour_id.to(device, non_blocking=True),
        contour_tok_id=b.contour_tok_id.to(device, non_blocking=True),
        axis_id=b.axis_id.to(device, non_blocking=True),
        axis_value=b.axis_value.to(device, non_blocking=True),
        axis_mask=b.axis_mask.to(device, non_blocking=True),
        deltas=b.deltas.to(device, non_blocking=True),
        pad_mask=b.pad_mask.to(device, non_blocking=True),
        lengths=b.lengths.to(device, non_blocking=True),
        meta=b.meta,
    )


class GlyphTupleDataset(Dataset):
    def __init__(self, items: List[Dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def _build_single_item_from_record(
    rec_text: str,
    origin: str,
    glyph_name: str,
    unicode_decimal: Optional[int],
    tup_idx: int,
    axis_spec: AxisSpec,
    only_axes: Set[str],
    allow_only_single_axis_tuples: bool,
    max_points_allowed: int,
    prediction_type: str,
    all_tuples: Optional[List[ParsedTuple]] = None,
) -> Optional[Dict]:

    from shared import TTGLYPH_FULL_RE, GVAR_RE

    m1 = TTGLYPH_FULL_RE.search(rec_text)
    m2 = GVAR_RE.search(rec_text)
    if not m1 or not m2:
        return None
    inner = (m1.group(2) or "").strip()
    gvar_inner = (m2.group(1) or "").strip()

    current_glyph = parse_ttglyph_inner(inner)
    if all_tuples is None:
        all_tuples = parse_gvar_inner(gvar_inner)

    if tup_idx >= len(all_tuples):
        return None

    base_points_in_glyph = len(current_glyph.points_xy)
    if base_points_in_glyph > max_points_allowed:
        return None

    tup = all_tuples[tup_idx]
    if not tup.coords:
        return None

    coord_axes = list(tup.coords.keys())
    coord_axes_set = set(coord_axes)
    if only_axes:
        if not coord_axes_set.issubset(only_axes):
            return None
    if allow_only_single_axis_tuples and len(coord_axes) != 1:
        return None

    axis_names = list(coord_axes)
    axis_values = [float(tup.coords[a]["value"]) for a in axis_names]

    points_xy = list(current_glyph.points_xy)
    oncurve = list(current_glyph.oncurve)
    contour_id = list(current_glyph.contour_id)

    if prediction_type == "tuple":
        deltas_sparse = [(int(pt), float(dx), float(dy)) for (pt, dx, dy) in tup.deltas]
    elif prediction_type == "total":
        axis_values_dict = {
            axis_names[i]: axis_values[i] for i in range(len(axis_names))
        }
        num_points = len(points_xy) + 4
        accumulated_dx = [0.0] * num_points
        accumulated_dy = [0.0] * num_points
        for other_tup in all_tuples:
            weight = compute_tuple_weight(other_tup.coords, axis_values_dict)
            for pt, dx, dy in other_tup.deltas:
                pt = int(pt)
                if 0 <= pt < num_points:
                    accumulated_dx[pt] += weight * float(dx)
                    accumulated_dy[pt] += weight * float(dy)
        deltas_sparse = [
            (pt, accumulated_dx[pt], accumulated_dy[pt])
            for pt in range(num_points)
            if abs(accumulated_dx[pt]) > 1e-6 or abs(accumulated_dy[pt]) > 1e-6
        ]
    elif prediction_type == "residual":
        current_axes_set = set(axis_names)
        axis_values_dict = {
            axis_names[i]: axis_values[i] for i in range(len(axis_names))
        }
        num_points = len(points_xy) + 4
        accumulated_dx = [0.0] * num_points
        accumulated_dy = [0.0] * num_points
        for other_tup_idx, other_tup in enumerate(all_tuples):
            if other_tup_idx == tup_idx:
                continue
            if not other_tup.coords:
                continue
            other_axes_set = set(other_tup.coords.keys())
            if not (
                other_axes_set.issubset(current_axes_set)
                and other_axes_set != current_axes_set
            ):
                continue
            weight = compute_tuple_weight(other_tup.coords, axis_values_dict)
            for pt, dx, dy in other_tup.deltas:
                pt = int(pt)
                if 0 <= pt < num_points:
                    accumulated_dx[pt] += weight * float(dx)
                    accumulated_dy[pt] += weight * float(dy)
        points_xy = [
            (
                float(points_xy[i][0]) + accumulated_dx[i],
                float(points_xy[i][1]) + accumulated_dy[i],
            )
            for i in range(len(points_xy))
        ]
        deltas_sparse = [(int(pt), float(dx), float(dy)) for (pt, dx, dy) in tup.deltas]
    else:
        raise ValueError(f"Unknown prediction_type: {prediction_type}")

    item = dict(
        points_xy=points_xy,
        oncurve=oncurve,
        contour_id=contour_id,
        deltas_sparse=deltas_sparse,
        axis_names=axis_names,
        axis_values=[float(v) for v in axis_values],
        origin=origin,
        glyph_name=glyph_name,
        unicode_decimal=unicode_decimal,
        glyph_id=(origin, glyph_name),
    )
    item["axis_ids"] = [axis_spec.name_to_id[a] for a in axis_names]
    item["axis_values_vec"] = [float(v) for v in axis_values]
    item["axis_id"] = int(item["axis_ids"][0])
    item["axis_value"] = float(item["axis_values_vec"][0])
    item["base_N"] = len(item["points_xy"])
    return item


def build_regression_index(
    data_dir: Path,
    only_axes: Set[str],
    allow_only_single_axis_tuples: bool,
    max_points_allowed: int,
    limit_examples: int,
    prediction_type: str,
) -> Tuple[List[Dict], AxisSpec, int, int, Dict[Tuple[str, str], Tuple]]:

    from shared import (
        GLYPH_RECORD_RE,
        TTGLYPH_FULL_RE,
        GVAR_RE,
        UNICODE_DEC_RE,
    )

    axis_names_seen: Set[str] = set()
    index_items: List[Dict] = []
    n_tuples = 0
    max_contour_id_seen = -1
    max_points_in_glyph = -1
    glyphs_skipped_due_to_max_points = 0
    total_glyphs_seen = 0

    glyph_data: Dict[Tuple[str, str], Tuple] = {}

    paths = sorted(list(data_dir.rglob("*.xml")) + list(data_dir.rglob("*.xml.gz")))
    seen_names: Set[str] = set()
    for p in paths:
        if p.name in seen_names:
            print(f"[ERROR] Duplicate font name found: {p.name}")
            exit(1)
        seen_names.add(p.name)

    for font_xml_path in tqdm.tqdm(
        paths, desc="Indexing font files (lazy) into tuple metadata"
    ):
        try:
            txt = font_xml_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[skip] {font_xml_path}: {e}")
            continue

        for rec_m in GLYPH_RECORD_RE.finditer(txt):
            rec = rec_m.group(1)
            m1 = TTGLYPH_FULL_RE.search(rec)
            m2 = GVAR_RE.search(rec)
            if not m1 or not m2:
                continue

            attrs = m1.group(1) or ""
            inner = (m1.group(2) or "").strip()
            gvar_inner = (m2.group(1) or "").strip()

            name_m = re.search(r'name="([^"]+)"', attrs)
            glyph_name = name_m.group(1) if name_m else ""

            uni_m = UNICODE_DEC_RE.search(rec)
            if not uni_m:
                continue
            unicode_decimal = int(uni_m.group(1))

            total_glyphs_seen += 1

            current_glyph = parse_ttglyph_inner(inner)
            tuples_of_this_glyph = parse_gvar_inner(gvar_inner)

            base_points_in_glyph = len(current_glyph.points_xy)
            if base_points_in_glyph > max_points_in_glyph:
                max_points_in_glyph = base_points_in_glyph
            if base_points_in_glyph > max_points_allowed:
                glyphs_skipped_due_to_max_points += 1
                continue

            if current_glyph.contour_id:
                try:
                    max_contour_id_seen = max(
                        max_contour_id_seen, int(max(current_glyph.contour_id))
                    )
                except Exception:
                    pass

            glyph_key = (str(font_xml_path), glyph_name)

            glyph_data[glyph_key] = None

            rec_start = rec_m.start()
            rec_end = rec_m.end()

            for tup_idx, tup in enumerate(tuples_of_this_glyph):
                if not tup.coords:
                    continue
                coord_axes = list(tup.coords.keys())
                coord_axes_set = set(coord_axes)
                if only_axes and not coord_axes_set.issubset(only_axes):
                    continue
                if allow_only_single_axis_tuples and len(coord_axes) != 1:
                    continue

                axis_names = list(coord_axes)
                axis_values = [float(tup.coords[a]["value"]) for a in axis_names]
                axis_names_seen.update(axis_names)

                index_items.append(
                    dict(
                        file_path=str(font_xml_path),
                        record_start=rec_start,
                        record_end=rec_end,
                        tup_idx=tup_idx,
                        axis_names=axis_names,
                        axis_values=axis_values,
                        origin=str(font_xml_path),
                        glyph_name=glyph_name,
                        unicode_decimal=unicode_decimal,
                        glyph_id=(str(font_xml_path), glyph_name),
                        base_N=base_points_in_glyph,
                    )
                )
                n_tuples += 1
                if limit_examples > 0 and n_tuples >= limit_examples:
                    break
            if limit_examples > 0 and n_tuples >= limit_examples:
                break
        if limit_examples > 0 and n_tuples >= limit_examples:
            break

    if not index_items:
        raise RuntimeError(
            "No regression items found. Check --only-axes / tuple filtering."
        )

    axis_spec = AxisSpec.from_axis_list(list(axis_names_seen))

    for it in index_items:
        it["axis_ids"] = [axis_spec.name_to_id[a] for a in it["axis_names"]]
        it["axis_values_vec"] = [float(v) for v in it["axis_values"]]
        it["axis_id"] = int(it["axis_ids"][0])
        it["axis_value"] = float(it["axis_values_vec"][0])

    max_points = max(it["base_N"] + 4 for it in index_items)

    total_fonts_len = len(list(data_dir.rglob("*.xml")))
    num_fonts_used = len(sorted({it["origin"] for it in index_items}))
    print(
        f"[data-lazy] total font files in folder: {total_fonts_len}, font files used: {num_fonts_used}"
    )
    print(f"[data-lazy] tuples/items indexed: {len(index_items)}")
    print(f"[data-lazy] axes seen: {axis_spec.id_to_name}")
    print(f"[data-lazy] max_points (incl phantom): {max_points}")
    print(f"[data-lazy] max_contour_id_seen: {max_contour_id_seen}")
    print(
        f"[data-lazy] max base points in a glyph contour (no phantom): {max_points_in_glyph}"
    )
    if glyphs_skipped_due_to_max_points > 0:
        print(
            f"[data-lazy] {glyphs_skipped_due_to_max_points} glyphs skipped (max_points_allowed={max_points_allowed}) "
            f"out of {total_glyphs_seen} total"
        )

    return index_items, axis_spec, max_points, max_contour_id_seen, glyph_data


class LazyGlyphTupleDataset(Dataset):

    def __init__(
        self,
        index_items: List[Dict],
        axis_spec: AxisSpec,
        only_axes: Set[str],
        allow_only_single_axis_tuples: bool,
        max_points_allowed: int,
        prediction_type: str,
    ):
        self.index = index_items
        self.axis_spec = axis_spec
        self.only_axes = only_axes
        self.allow_only_single_axis_tuples = allow_only_single_axis_tuples
        self.max_points_allowed = max_points_allowed
        self.prediction_type = prediction_type

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        meta = self.index[idx]
        fpath = meta["file_path"]
        txt = Path(fpath).read_text(encoding="utf-8")
        rec_text = txt[meta["record_start"] : meta["record_end"]]

        item = _build_single_item_from_record(
            rec_text=rec_text,
            origin=meta["origin"],
            glyph_name=meta["glyph_name"],
            unicode_decimal=meta["unicode_decimal"],
            tup_idx=meta["tup_idx"],
            axis_spec=self.axis_spec,
            only_axes=self.only_axes,
            allow_only_single_axis_tuples=self.allow_only_single_axis_tuples,
            max_points_allowed=self.max_points_allowed,
            prediction_type=self.prediction_type,
        )
        if item is None:

            item = dict(
                points_xy=[],
                oncurve=[],
                contour_id=[],
                deltas_sparse=[],
                axis_names=meta["axis_names"],
                axis_values=meta["axis_values"],
                origin=meta["origin"],
                glyph_name=meta["glyph_name"],
                unicode_decimal=meta["unicode_decimal"],
                glyph_id=meta["glyph_id"],
                axis_ids=meta["axis_ids"],
                axis_values_vec=meta["axis_values_vec"],
                axis_id=meta["axis_id"],
                axis_value=meta["axis_value"],
                base_N=0,
            )
        if "line_constraints" in meta:
            item["line_constraints"] = meta.get("line_constraints")
        return item


def build_regression_items(
    data_dir: Path,
    only_axes: Set[str],
    allow_only_single_axis_tuples: bool,
    max_points_allowed: int,
    limit_examples: int,
    prediction_type: str,
) -> Tuple[List[Dict], AxisSpec, int, int, Dict[Tuple[str, str], Tuple]]:

    axis_names_seen = set()
    raw_items: List[Dict] = []
    n_tuples = 0
    max_contour_id_seen = -1
    max_points_in_glyph = -1
    glyphs_skipped_due_to_max_points = 0
    total_glyphs_seen = 0

    glyph_data: Dict[Tuple[str, str], Tuple[ParsedGlyph, List[ParsedTuple]]] = {}

    cache_file = data_dir / f".glyph_cache_maxpts{max_points_allowed}.pkl"
    cache_loaded = False

    if cache_file.exists():
        try:
            print(
                f"[glyph-cache] loading from {cache_file} ({cache_file.stat().st_size / 1e6:.1f} MB) ..."
            )
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            glyph_data = cached["glyph_data"]
            max_contour_id_seen = cached["max_contour_id_seen"]
            max_points_in_glyph = cached["max_points_in_glyph"]
            glyphs_skipped_due_to_max_points = cached[
                "glyphs_skipped_due_to_max_points"
            ]
            total_glyphs_seen = cached["total_glyphs_seen"]
            cache_loaded = True
            print(
                f"[glyph-cache] loaded {len(glyph_data)} glyphs from {cache_file} ({cache_file.stat().st_size / 1e6:.1f} MB)"
            )
        except Exception as e:
            print(f"[glyph-cache] WARNING: failed to load cache ({e}), will re-parse")

    if not cache_loaded:
        debugsmallamount = False
        if debugsmallamount:
            print("")
            print("warning, debugsmallamount is True - REMOVE ME +++++++++++")
            print("")

        amount_processed = 0
        for src_inner, gvar_inner, origin, glyph_name, unicode_decimal in tqdm.tqdm(
            iter_all_pairs_with_glyph_name(data_dir),
            desc="Iterating font files and processing glyphs [build_regression_items]",
        ):
            total_glyphs_seen += 1
            amount_processed += 1
            if debugsmallamount and amount_processed > 70000:
                break
            current_glyph = parse_ttglyph_inner(src_inner)
            tuples_of_this_glyph = parse_gvar_inner(gvar_inner)

            base_points_in_glyph = len(current_glyph.points_xy)
            if base_points_in_glyph > max_points_in_glyph:
                max_points_in_glyph = base_points_in_glyph

            if base_points_in_glyph > max_points_allowed:
                glyphs_skipped_due_to_max_points += 1
                continue

            if current_glyph.contour_id:
                try:
                    max_contour_id_seen = max(
                        max_contour_id_seen, int(max(current_glyph.contour_id))
                    )
                except Exception:
                    pass

            glyph_key = (origin, glyph_name)
            glyph_data[glyph_key] = (
                current_glyph,
                tuples_of_this_glyph,
                unicode_decimal,
            )

            n_tuples += len(tuples_of_this_glyph)
            if limit_examples > 0 and n_tuples >= limit_examples:
                break

        if limit_examples == 0:
            try:
                print(
                    f"[glyph-cache] saving {len(glyph_data)} glyphs to {cache_file} ..."
                )
                tmp = cache_file.with_suffix(".pkl.tmp")
                with open(tmp, "wb") as f:
                    pickle.dump(
                        {
                            "glyph_data": glyph_data,
                            "max_contour_id_seen": max_contour_id_seen,
                            "max_points_in_glyph": max_points_in_glyph,
                            "glyphs_skipped_due_to_max_points": glyphs_skipped_due_to_max_points,
                            "total_glyphs_seen": total_glyphs_seen,
                        },
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                tmp.rename(cache_file)
                print(
                    f"[glyph-cache] saved {len(glyph_data)} glyphs to {cache_file} ({cache_file.stat().st_size / 1e6:.1f} MB)"
                )
            except Exception as e:
                print(f"[glyph-cache] WARNING: failed to save cache ({e})")

    for (origin, glyph_name), (
        current_glyph,
        tuples_of_this_glyph,
        unicode_decimal,
    ) in tqdm.tqdm(
        glyph_data.items(),
        desc="Building training items and calculate labels. (out of glyphs)",
    ):

        for tup_idx, tup in enumerate(tuples_of_this_glyph):
            if not tup.coords:
                continue

            if False:

                if "Truculenta" in origin and glyph_name == "one":

                    axis_values_check = {
                        axis: float(tup.coords[axis]["value"]) for axis in tup.coords
                    }
                    if abs(axis_values_check.get("wdth", 0) - (1)) < 1e-6:
                        pass
                        print(
                            f"**** Debug breakpoint: origin={origin}, glyph={glyph_name}, tuple index={tup_idx}, coords={tup.coords}"
                        )

            coord_axes = list(tup.coords.keys())

            coord_axes_set = set(coord_axes)
            if only_axes:

                if not coord_axes_set.issubset(only_axes):
                    continue

            if allow_only_single_axis_tuples:

                if len(coord_axes) != 1:
                    continue

            axis_names = list(coord_axes)
            axis_values = [float(tup.coords[a]["value"]) for a in axis_names]

            axis_names_seen.update(axis_names)

            points_xy = list(current_glyph.points_xy)
            oncurve = list(current_glyph.oncurve)
            contour_id = list(current_glyph.contour_id)

            if prediction_type == "tuple":

                deltas_sparse = [
                    (int(pt), float(dx), float(dy)) for (pt, dx, dy) in tup.deltas
                ]
            elif prediction_type == "total":

                axis_values_dict = {
                    axis_names[i]: axis_values[i] for i in range(len(axis_names))
                }

                num_points = len(points_xy) + 4
                accumulated_dx = [0.0] * num_points
                accumulated_dy = [0.0] * num_points

                for other_tup in tuples_of_this_glyph:

                    tuple_coords = other_tup.coords

                    weight = compute_tuple_weight(tuple_coords, axis_values_dict)
                    pass

                    for pt, dx, dy in other_tup.deltas:
                        pt = int(pt)

                        if not (0 <= pt < num_points):
                            raise ValueError(
                                f"Point index {pt} out of bounds for glyph '{glyph_name}' "
                                f"with {num_points} points (outline={len(points_xy)}, phantom=4). "
                                f"Origin: {origin}"
                            )

                        accumulated_dx[pt] += weight * float(dx)
                        accumulated_dy[pt] += weight * float(dy)

                deltas_sparse = [
                    (pt, accumulated_dx[pt], accumulated_dy[pt])
                    for pt in range(num_points)
                    if abs(accumulated_dx[pt]) > 1e-6 or abs(accumulated_dy[pt]) > 1e-6
                ]
            elif prediction_type == "residual":

                current_axes_set = set(axis_names)

                axis_values_dict = {
                    axis_names[i]: axis_values[i] for i in range(len(axis_names))
                }

                num_points = len(points_xy) + 4
                accumulated_dx = [0.0] * num_points
                accumulated_dy = [0.0] * num_points

                for other_tup_idx, other_tup in enumerate(tuples_of_this_glyph):

                    if other_tup_idx == tup_idx:
                        continue

                    if not other_tup.coords:
                        continue

                    other_axes_set = set(other_tup.coords.keys())
                    if not (
                        other_axes_set.issubset(current_axes_set)
                        and other_axes_set != current_axes_set
                    ):
                        continue

                    tuple_coords = other_tup.coords

                    weight = compute_tuple_weight(tuple_coords, axis_values_dict)

                    for pt, dx, dy in other_tup.deltas:
                        pt = int(pt)
                        if not (0 <= pt < num_points):
                            raise ValueError(
                                f"Point index {pt} out of bounds for glyph '{glyph_name}' "
                                f"with {num_points} points (outline={len(points_xy)}, phantom=4). "
                                f"Origin: {origin}"
                            )
                        accumulated_dx[pt] += weight * float(dx)
                        accumulated_dy[pt] += weight * float(dy)

                points_xy = [
                    (
                        float(points_xy[i][0]) + accumulated_dx[i],
                        float(points_xy[i][1]) + accumulated_dy[i],
                    )
                    for i in range(len(points_xy))
                ]

                deltas_sparse = [
                    (int(pt), float(dx), float(dy)) for (pt, dx, dy) in tup.deltas
                ]
            else:
                raise ValueError(f"Unknown prediction_type: {prediction_type}")

            raw_items.append(
                dict(
                    points_xy=points_xy,
                    oncurve=oncurve,
                    contour_id=contour_id,
                    deltas_sparse=deltas_sparse,
                    axis_names=axis_names,
                    axis_values=[float(v) for v in axis_values],
                    origin=origin,
                    glyph_name=glyph_name,
                    unicode_decimal=unicode_decimal,
                    glyph_id=(origin, glyph_name),
                )
            )

            n_tuples += 1
            if limit_examples > 0 and n_tuples >= limit_examples:
                break

        if limit_examples > 0 and n_tuples >= limit_examples:
            break

    if not raw_items:
        raise RuntimeError(
            "No regression items found. Check --only-axes / tuple filtering."
        )

    axis_spec = AxisSpec.from_axis_list(list(axis_names_seen))

    max_points = max(len(it["points_xy"]) + 4 for it in raw_items)

    for it in raw_items:
        it["axis_ids"] = [axis_spec.name_to_id[a] for a in it["axis_names"]]
        it["axis_values_vec"] = [float(v) for v in it["axis_values"]]

        it["axis_id"] = int(it["axis_ids"][0])
        it["axis_value"] = float(it["axis_values_vec"][0])
        it["base_N"] = len(it["points_xy"])

    total_fonts_len = len(list(data_dir.rglob("*.xml")))
    num_fonts_used = len(sorted({it["origin"] for it in raw_items}))

    print(
        f"[data] total font files in folder: {total_fonts_len}, font files used for tuples: {num_fonts_used}"
    )
    print(f"[data] tuples/items: {len(raw_items)}")
    print(f"[data] axes seen in dataset: {axis_spec.id_to_name}")
    print(f"[data] max_points (incl phantom): {max_points}")
    print(f"[data] max_contour_id_seen: {max_contour_id_seen}")
    print(
        f"[data] max base points in a glyph contour (no phantom): {max_points_in_glyph}"
    )
    if glyphs_skipped_due_to_max_points > 0:
        print(
            f"[data] {glyphs_skipped_due_to_max_points} glyphs were skipped due to max_points_allowed={max_points_allowed} "
            f"out of {total_glyphs_seen} glyphs in total"
        )

    glyph_data = {k: None for k in glyph_data}

    return raw_items, axis_spec, max_points, max_contour_id_seen, glyph_data


class RegressionCollator:

    def __init__(
        self,
        point_contour_info: int,
        max_contour_id_seen_plus_one: int,
        device: Optional[torch.device] = None,
    ):
        self.device = device
        self.point_contour_info = int(point_contour_info)
        self.max_contour_id_seen_plus_one = int(max_contour_id_seen_plus_one)

        self.phantom_token_id = self.max_contour_id_seen_plus_one
        self.pad_token_id = self.max_contour_id_seen_plus_one + 1

    def point_feat_dim(self) -> int:

        if self.point_contour_info == 2:
            return 4
        if self.point_contour_info == 3:
            return 4
        return 3

    def __call__(self, batch: List[Dict]) -> RegressionBatch:
        B = len(batch)

        points_lengths = torch.tensor(
            [len(b["points_xy"]) + 4 for b in batch], dtype=torch.long
        )
        max_points_length = int(points_lengths.max().item())

        point_dimension = self.point_feat_dim()
        points_batch = torch.zeros(
            (B, max_points_length, point_dimension), dtype=torch.float32
        )

        contour_id = torch.full((B, max_points_length), -1, dtype=torch.long)
        contour_token_id = torch.full(
            (B, max_points_length), self.pad_token_id, dtype=torch.long
        )

        deltas = torch.zeros((B, max_points_length, 2), dtype=torch.float32)

        axis_length_per_example = [len(example.get("axis_ids")) for example in batch]
        max_axis_length_seen_in_batch = (
            max(axis_length_per_example) if axis_length_per_example else 1
        )
        axis_id = torch.zeros((B, max_axis_length_seen_in_batch), dtype=torch.long)
        axis_value = torch.zeros(
            (B, max_axis_length_seen_in_batch), dtype=torch.float32
        )
        axis_mask = torch.zeros((B, max_axis_length_seen_in_batch), dtype=torch.bool)

        for example_id, example in enumerate(batch):
            axis_id_list_for_this_example = example.get(
                "axis_ids", [int(example["axis_id"])]
            )
            axis_val_list_for_this_example = example.get(
                "axis_values_vec", [float(example["axis_value"])]
            )
            k = min(len(axis_id_list_for_this_example), max_axis_length_seen_in_batch)
            if k > 0:
                axis_id[example_id, :k] = torch.tensor(
                    axis_id_list_for_this_example[:k], dtype=torch.long
                )
                axis_value[example_id, :k] = torch.tensor(
                    axis_val_list_for_this_example[:k], dtype=torch.float32
                )
                axis_mask[example_id, :k] = True

        pad_mask = torch.ones((B, max_points_length), dtype=torch.bool)
        meta: List[Dict] = []

        coord_scale = 2.0 / float(SOURCE_UPM)

        for example_id, example in enumerate(batch):
            number_points_in_example = len(example["points_xy"])
            number_points_in_example_including_phantom = number_points_in_example + 4

            if len(example["points_xy"]) == 0:
                example_points = torch.empty((0, 2), dtype=torch.float32)
            else:
                example_points = torch.tensor(example["points_xy"], dtype=torch.float32)

            oncurve_indicator_per_point = torch.tensor(
                example["oncurve"], dtype=torch.float32
            ).view(number_points_in_example, 1)

            if len(oncurve_indicator_per_point) != number_points_in_example:
                raise RuntimeError(
                    f"Shape mismatch in example {example_id}: "
                    f"len(points_xy)={number_points_in_example} but len(oncurve)={len(oncurve_indicator_per_point)}. "
                    f"Origin: {example.get('origin')}, Glyph: {example.get('glyph_name')}"
                )

            example_points_normalized = example_points * coord_scale - 1.0
            pts_base_3 = torch.cat(
                [example_points_normalized, oncurve_indicator_per_point], dim=-1
            )

            if pts_base_3.shape[1] != 3:
                raise RuntimeError(
                    f"Shape error in example {example_id}: pts_base_3 has shape {pts_base_3.shape}, expected second dimension to be 3. "
                    f"Origin: {example.get('origin')}, Glyph: {example.get('glyph_name')}"
                )

            phantom_xy = torch.zeros((4, 2), dtype=torch.float32)
            phantom_oc = torch.full((4, 1), -1.0, dtype=torch.float32)
            pts_ph_3 = torch.cat([phantom_xy, phantom_oc], dim=-1)

            pts3 = torch.cat([pts_base_3, pts_ph_3], dim=0)

            contour_ids_per_base_point = torch.tensor(
                example["contour_id"], dtype=torch.long
            )
            phantom_contour_ids = torch.tensor(
                [-(k + 2) for k in range(4)], dtype=torch.long
            )
            contour_ids = torch.cat(
                [contour_ids_per_base_point, phantom_contour_ids], dim=0
            )

            tok = torch.empty(
                (number_points_in_example_including_phantom,), dtype=torch.long
            )
            if self.max_contour_id_seen_plus_one > 0:

                tok[:number_points_in_example] = torch.clamp(
                    contour_ids_per_base_point,
                    min=0,
                    max=self.max_contour_id_seen_plus_one - 1,
                )
            else:
                tok[:number_points_in_example] = 0
            tok[number_points_in_example:] = self.phantom_token_id

            deltas_for_this_example = torch.zeros(
                (number_points_in_example_including_phantom, 2), dtype=torch.float32
            )
            for pt, dx, dy in example.get("deltas_sparse"):
                if 0 <= pt < number_points_in_example_including_phantom:
                    deltas_for_this_example[pt, 0] = float(dx)
                    deltas_for_this_example[pt, 1] = float(dy)

            deltas_for_this_example = deltas_for_this_example * coord_scale

            if self.point_contour_info in (2, 3):
                extra = torch.zeros(
                    (number_points_in_example_including_phantom, 1), dtype=torch.float32
                )
                if number_points_in_example > 0:
                    if self.point_contour_info == 2:

                        extra[0, 0] = 1.0
                        if number_points_in_example > 1:
                            same_prev = (
                                contour_ids_per_base_point[1:]
                                == contour_ids_per_base_point[:-1]
                            )
                            extra[1:number_points_in_example, 0] = (~same_prev).float()
                        extra[number_points_in_example:, 0] = 0.0
                    else:

                        pos = torch.zeros(
                            (number_points_in_example,), dtype=torch.float32
                        )

                        start = 0
                        while start < number_points_in_example:
                            contour_id_for_this_point = int(
                                contour_ids_per_base_point[start].item()
                            )
                            end = start + 1
                            while (
                                end < number_points_in_example
                                and int(contour_ids_per_base_point[end].item())
                                == contour_id_for_this_point
                            ):
                                end += 1
                            number_of_points_in_this_contour = end - start
                            denom = float(max(1, number_of_points_in_this_contour - 1))
                            pos[start:end] = (
                                torch.arange(
                                    number_of_points_in_this_contour,
                                    dtype=torch.float32,
                                )
                                / denom
                            )
                            start = end
                        extra[:number_points_in_example, 0] = pos
                        extra[number_points_in_example:, 0] = 0.0

                pts = torch.cat([pts3, extra], dim=-1)
            else:
                pts = pts3

            points_batch[
                example_id, :number_points_in_example_including_phantom, : pts.shape[1]
            ] = pts
            contour_id[example_id, :number_points_in_example_including_phantom] = (
                contour_ids
            )
            contour_token_id[
                example_id, :number_points_in_example_including_phantom
            ] = tok
            deltas[example_id, :number_points_in_example_including_phantom] = (
                deltas_for_this_example
            )
            pad_mask[example_id, :number_points_in_example_including_phantom] = False

            meta.append(
                {
                    "origin": example.get("origin"),
                    "glyph_name": example.get("glyph_name"),
                    "unicode_decimal": example.get("unicode_decimal"),
                    "axis_names": list(example.get("axis_names")),
                    "axis_values": list(
                        example.get("axis_values", example.get("axis_values_vec"))
                    ),
                    "base_N": int(example.get("base_N", number_points_in_example)),
                    "line_constraints": list(example.get("line_constraints", [])),
                }
            )

        if self.device is not None:
            points_batch = points_batch.to(self.device)
            contour_id = contour_id.to(self.device)
            contour_token_id = contour_token_id.to(self.device)
            axis_id = axis_id.to(self.device)
            axis_value = axis_value.to(self.device)
            axis_mask = axis_mask.to(self.device)
            deltas = deltas.to(self.device)
            pad_mask = pad_mask.to(self.device)
            points_lengths = points_lengths.to(self.device)

        return RegressionBatch(
            points=points_batch,
            contour_id=contour_id,
            contour_tok_id=contour_token_id,
            axis_id=axis_id,
            axis_value=axis_value,
            axis_mask=axis_mask,
            deltas=deltas,
            pad_mask=pad_mask,
            lengths=points_lengths,
            meta=meta,
        )


def _denorm_xy_to_upm(xy_norm: torch.Tensor) -> torch.Tensor:

    return (xy_norm + 1.0) * (SOURCE_UPM / 2.0)


def _denorm_delta_to_upm(dxy_norm: torch.Tensor) -> torch.Tensor:

    return dxy_norm * (SOURCE_UPM / 2.0)


@torch.no_grad()
def evaluate(
    args,
    model: Optional[nn.Module],
    test_loader: DataLoader,
    device: torch.device,
    use_bf16: bool,
    *,
    collator: Optional[RegressionCollator] = None,
    print_sample: bool = True,
    sample_k: int = 1,
) -> float:
    line_preservation_enabled = bool(
        getattr(args, "linepreservationatinference", False)
    )
    if not args.nullbaseline:
        if model is None:
            raise ValueError("Model is required unless --nullbaseline is set.")
        model.eval()

    def _predict_batch(b: RegressionBatch) -> torch.Tensor:
        if args.nullbaseline:

            return torch.zeros_like(b.deltas)

        if not args.only_eval_no_high_order:
            return model(
                b.points,
                b.contour_tok_id,
                b.axis_id,
                b.axis_value,
                b.pad_mask,
                b.axis_mask,
                combinatorial_max_order=args.combinatorial_max_order,
            )

        B, K = b.axis_mask.shape
        pred_sum: Optional[torch.Tensor] = None
        for k in range(K):
            if not bool(b.axis_mask[:, k].any().item()):
                continue
            axis_mask_single = torch.zeros(
                (B, K), dtype=b.axis_mask.dtype, device=b.axis_mask.device
            )
            axis_mask_single[:, k] = b.axis_mask[:, k]
            pred_k = model(
                b.points,
                b.contour_tok_id,
                b.axis_id,
                b.axis_value,
                b.pad_mask,
                axis_mask_single,
                combinatorial_max_order=args.combinatorial_max_order,
            )
            pred_sum = pred_k if pred_sum is None else (pred_sum + pred_k)

        if pred_sum is None:
            return model(
                b.points,
                b.contour_tok_id,
                b.axis_id,
                b.axis_value,
                b.pad_mask,
                b.axis_mask,
                combinatorial_max_order=args.combinatorial_max_order,
            )
        return pred_sum

    if (
        print_sample
        and (not args.only_eval)
        and collator is not None
        and len(test_loader.dataset) > 0
    ):
        if True:
            i = random.randrange(len(test_loader.dataset))
        else:

            i = 0
            for j in range(len(test_loader.dataset)):
                ex = test_loader.dataset[j]
                if (
                    ex.get("origin", "").find("ScienceGothic") >= 0
                    and ex.get("glyph_name") == "M"
                ):
                    axis_names = ex.get("axis_names", [])
                    axis_values = ex.get("axis_values", [])
                    axis_dict = {n: v for n, v in zip(axis_names, axis_values)}
                    if (
                        abs(axis_dict.get("slnt", 0) + 1) < 1e-6
                        and abs(axis_dict.get("wdth", 0) + 1) < 1e-6
                        and abs(axis_dict.get("wght", 0) + 1) < 1e-6
                    ):
                        i = j
                        break
        ex = test_loader.dataset[i]
        b1 = collator([ex])
        b1 = move_batch_to_device(b1, device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(use_bf16 and device.type == "cuda"),
        ):

            p1 = _predict_batch(b1)

        meta = b1.meta[0]
        origin = meta.get("origin")
        glyph_name = meta.get("glyph_name")

        axis_names = meta.get("axis_names")
        axis_values = meta.get("axis_values")

        base_N = int(meta.get("base_N", 0))

        valid_mask = ~b1.pad_mask[0]
        pts = b1.points[0][valid_mask]
        cids = b1.contour_id[0][valid_mask]
        tgt = b1.deltas[0][valid_mask]
        pred = p1[0][valid_mask]

        pts_upm = _denorm_xy_to_upm(pts[:, :2])
        tgt_upm = _denorm_delta_to_upm(tgt)
        pred_upm = _denorm_delta_to_upm(pred)

        N = pts_upm.shape[0]
        sample_k_min = max(1, min(int(sample_k), N))
        print("\n" + "=" * 100)
        print("[eval sample]")
        print(f"  origin     : {origin}")
        print(f"  glyph_name : {glyph_name}")

        if axis_names and axis_values and len(axis_names) == len(axis_values):
            axis_str = ", ".join(f"{n}={v:g}" for n, v in zip(axis_names, axis_values))
        else:
            axis_str = "(missing axis info)"

        print(f"  axes       : {axis_str}")

        print(f"  N_total    : {N} (base_N={base_N}, phantom={max(0, N-base_N)})")
        print("  showing first k points in integer UPM units:")
        for j in range(sample_k_min):
            x, y = pts_upm[j].tolist()
            oc = float(pts[j, 2].item())
            dx_t, dy_t = tgt_upm[j].tolist()
            dx_p, dy_p = pred_upm[j].tolist()
            cid = int(cids[j].item())
            tag = "base   " if (base_N > 0 and j < base_N) else "phantom"

            print(
                f"    cont={cid:2d}  pt={j:4d} [{tag}] x={x:5.0f} y={y:5.0f} oncurve={oc:5.0f} | "
                f"tgt=({dx_t:5.0f},{dy_t:5.0f}) pred=({dx_p:5.0f},{dy_p:5.0f})"
            )
        print("=" * 100 + "\n")

    if line_preservation_enabled:
        print(
            "[warning] --linepreservationatinference enabled: eval will also report post-projected line-preserved predictions."
        )

    losses: List[float] = []
    corrected_losses: List[float] = []
    for batch in test_loader:
        b = move_batch_to_device(batch, device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(use_bf16 and device.type == "cuda"),
        ):

            pred = _predict_batch(b)

            eval_loss_name = args.eval_loss
            per = compute_loss(pred, b.deltas, eval_loss_name, args.huber_delta)
            per = per.masked_fill(b.pad_mask, 0.0)
            denom = (~b.pad_mask).sum().clamp(min=1)
            loss = per.sum() / denom

        losses.append(float(loss.detach().float().item()))

        if line_preservation_enabled:
            pred_corr = pred.detach().float().clone()
            for bi, meta in enumerate(b.meta):
                constraints = list(meta.get("line_constraints", []))
                if not constraints:
                    continue

                valid_mask = ~b.pad_mask[bi]
                n_valid = int(valid_mask.sum().item())
                if n_valid <= 0:
                    continue

                pts_abs = (
                    (b.points[bi, valid_mask, :2] + pred_corr[bi, valid_mask])
                    .detach()
                    .cpu()
                )
                pts_abs_list = [(float(x), float(y)) for x, y in pts_abs.tolist()]
                pts_corr_list = apply_line_preservation_inference(
                    pts_abs_list, constraints, move_anchors=True
                )
                pts_corr = torch.tensor(
                    pts_corr_list, dtype=pred_corr.dtype, device=pred_corr.device
                )
                pred_corr[bi, valid_mask] = (
                    pts_corr - b.points[bi, valid_mask, :2].detach().float()
                )

            per_corr = compute_loss(
                pred_corr, b.deltas, eval_loss_name, args.huber_delta
            )
            per_corr = per_corr.masked_fill(b.pad_mask, 0.0)
            loss_corr = per_corr.sum() / denom
            corrected_losses.append(float(loss_corr.detach().float().item()))

    if line_preservation_enabled and corrected_losses:
        mean_raw = float(sum(losses) / max(1, len(losses)))
        mean_corr = float(sum(corrected_losses) / max(1, len(corrected_losses)))
        if args.eval_loss == "mse":
            print(
                f"[eval linepreservationatinference] raw_mse={mean_raw:.6f} raw_rmse={math.sqrt(max(0.0, mean_raw)):.6f} "
                f"corrected_mse={mean_corr:.6f} corrected_rmse={math.sqrt(max(0.0, mean_corr)):.6f}"
            )
        else:
            print(
                f"[eval linepreservationatinference] raw_{args.eval_loss}={mean_raw:.6f} "
                f"corrected_{args.eval_loss}={mean_corr:.6f}"
            )

    return float(sum(losses) / max(1, len(losses)))


def _resolve_model_pt(load_checkpoint: Path) -> Path:
    p = Path(load_checkpoint)
    if p.is_dir():
        p = p / "model.pt"
    return p


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    only_axes_set = {a.strip() for a in args.only_axes.split(",") if a.strip()}

    hold_in_memory = args.hold_data_in_memory
    line_loss_alpha = None if args.lineloss is None else float(args.lineloss)
    line_loss_enabled = (line_loss_alpha is not None) and (line_loss_alpha > 0.0)

    if hold_in_memory:
        print("[data] Loading ALL data into RAM (default)")
        tuple_items, axis_spec, max_points, max_contour_id_seen, glyph_data = (
            build_regression_items(
                data_dir=args.data_dir,
                only_axes=only_axes_set,
                allow_only_single_axis_tuples=args.allow_only_single_axis_tuples,
                max_points_allowed=int(args.max_points_allowed),
                limit_examples=args.limit_examples,
                prediction_type=args.prediction_type,
            )
        )
    else:
        print(
            "[data] Building lightweight index (data will be read from disk on demand)"
        )
        tuple_items, axis_spec, max_points, max_contour_id_seen, glyph_data = (
            build_regression_index(
                data_dir=args.data_dir,
                only_axes=only_axes_set,
                allow_only_single_axis_tuples=args.allow_only_single_axis_tuples,
                max_points_allowed=int(args.max_points_allowed),
                limit_examples=args.limit_examples,
                prediction_type=args.prediction_type,
            )
        )

    rng = random.Random(args.seed)

    if args.split_method == "item":
        idxs = list(range(len(tuple_items)))
        rng.shuffle(idxs)
        cut = int(len(idxs) * args.train_frac)
        train_items = [tuple_items[i] for i in idxs[:cut]]
        test_items = [tuple_items[i] for i in idxs[cut:]]
        print(
            f"[split:item] train={len(train_items)} test={len(test_items)} (seed={args.seed})"
        )

    elif args.split_method == "glyphname":

        names = sorted({it.get("glyph_name") for it in tuple_items})
        names = [n for n in names if n != ""]
        if not names:
            raise RuntimeError("[split:glyphname] No glyph_name found in items.")

        rng.shuffle(names)
        cut = int(len(names) * args.train_frac)
        train_names = set(names[:cut])

        train_items = [it for it in tuple_items if it.get("glyph_name") in train_names]
        test_items = [
            it for it in tuple_items if it.get("glyph_name") not in train_names
        ]

        print(
            f"[split:glyphname] unique_glyphs total={len(names)} train={len(train_names)} test={len(names)-len(train_names)} "
            f"(train_frac={args.train_frac}, seed={args.seed})"
        )
        print(
            f"[split:glyphname] tuples train={len(train_items)} test={len(test_items)}"
        )

    elif args.split_method == "glyph":

        glyph_ids = sorted(glyph_data.keys())
        if not glyph_ids:
            raise RuntimeError("[split:glyph] No glyphs found in data.")

        rng.shuffle(glyph_ids)
        cut = int(len(glyph_ids) * args.train_frac)
        train_glyph_ids = set(glyph_ids[:cut])

        train_items = [
            it for it in tuple_items if it.get("glyph_id") in train_glyph_ids
        ]
        test_items = [
            it for it in tuple_items if it.get("glyph_id") not in train_glyph_ids
        ]

        print(
            f"[split:glyph] unique_glyphs total={len(glyph_ids)} train={len(train_glyph_ids)} test={len(glyph_ids)-len(train_glyph_ids)} "
            f"(train_frac={args.train_frac}, seed={args.seed})"
        )
        print(f"[split:glyph] tuples train={len(train_items)} test={len(test_items)}")

    elif args.split_method == "unicode":

        uvals = sorted(
            {
                it.get("unicode_decimal")
                for it in tuple_items
                if it.get("unicode_decimal") is not None
            }
        )
        if not uvals:
            raise RuntimeError(
                "[split:unicode] No unicode_decimal found in items. Check your XML or parsing."
            )

        fixed_split_dir = args.fixed_split_dir

        if fixed_split_dir is not None:

            fixed_split_dir = Path(fixed_split_dir)
            train_uni_txt = fixed_split_dir / "train_unicodes.txt"
            test_uni_txt = fixed_split_dir / "test_unicodes.txt"
            if not train_uni_txt.exists():
                raise FileNotFoundError(
                    f"[split:unicode] --fixed-split-dir: {train_uni_txt} not found"
                )
            if not test_uni_txt.exists():
                raise FileNotFoundError(
                    f"[split:unicode] --fixed-split-dir: {test_uni_txt} not found"
                )

            train_u_from_file = {
                int(line.strip())
                for line in train_uni_txt.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            test_u_from_file = {
                int(line.strip())
                for line in test_uni_txt.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }

            overlap = train_u_from_file & test_u_from_file
            if overlap:
                raise RuntimeError(
                    f"[split:unicode] train_unicodes.txt and test_unicodes.txt overlap on {len(overlap)} unicodes: "
                    f"{sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}"
                )

            uvals_set = set(uvals)
            train_u = train_u_from_file & uvals_set
            test_u = test_u_from_file & uvals_set

            missing_train = train_u_from_file - uvals_set
            missing_test = test_u_from_file - uvals_set
            if missing_train:
                print(
                    f"[split:unicode] WARNING: {len(missing_train)} unicodes in train_unicodes.txt not found in dataset (ignored)"
                )
            if missing_test:
                print(
                    f"[split:unicode] WARNING: {len(missing_test)} unicodes in test_unicodes.txt not found in dataset (ignored)"
                )

            all_listed = train_u | test_u
            unlisted = uvals_set - all_listed
            if unlisted:
                print(
                    f"[split:unicode] WARNING: {len(unlisted)} unicodes in dataset appear in neither "
                    f"train_unicodes.txt nor test_unicodes.txt — they will be EXCLUDED."
                )

            if not train_u or not test_u:
                raise RuntimeError(
                    f"[split:unicode] After matching, train_unicodes={len(train_u)} test_unicodes={len(test_u)}. "
                    f"Need at least 1 unicode in each split."
                )

            print(f"[split:unicode] Loaded fixed split from {fixed_split_dir}")
            print(
                f"[split:unicode] train_unicodes.txt: {len(train_u_from_file)} entries -> {len(train_u)} matched"
            )
            print(
                f"[split:unicode] test_unicodes.txt:  {len(test_u_from_file)} entries -> {len(test_u)} matched"
            )

        else:

            rng.shuffle(uvals)
            cut = int(len(uvals) * args.train_frac)
            train_u = set(uvals[:cut])
            test_u = set(uvals[cut:])

        train_items = [
            it
            for it in tuple_items
            if (it.get("unicode_decimal") in train_u)
            or (it.get("unicode_decimal") is None)
        ]
        test_items = [it for it in tuple_items if (it.get("unicode_decimal") in test_u)]

        args._train_unicodes = sorted(train_u)
        args._test_unicodes = sorted(test_u)

        print(
            f"[split:unicode] unique_unicodes total={len(uvals)} train={len(train_u)} test={len(test_u)} "
            f"(train_frac={args.train_frac}, seed={args.seed})"
        )
        print(
            f"[split:unicode] tuples train={len(train_items)} test={len(test_items)} (note: unicode=None kept in train)"
        )

    elif args.split_method == "font":

        rng = random.Random(args.seed)

        fonts = sorted({it.get("origin", "") for it in tuple_items})
        fonts = [f for f in fonts if f != ""]
        if not fonts:
            raise RuntimeError("[split:font] No origin found in items.")

        fixed_split_dir = args.fixed_split_dir

        if fixed_split_dir is not None:

            fixed_split_dir = Path(fixed_split_dir)
            train_txt = fixed_split_dir / "train.txt"
            test_txt = fixed_split_dir / "test.txt"
            if not train_txt.exists():
                raise FileNotFoundError(
                    f"[split:font] --fixed-split-dir: {train_txt} not found"
                )
            if not test_txt.exists():
                raise FileNotFoundError(
                    f"[split:font] --fixed-split-dir: {test_txt} not found"
                )

            train_basenames = {
                Path(line).name
                for line in train_txt.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            test_basenames = {
                Path(line).name
                for line in test_txt.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }

            overlap = train_basenames & test_basenames
            if overlap:
                raise RuntimeError(
                    f"[split:font] train.txt and test.txt overlap on {len(overlap)} fonts: "
                    f"{sorted(overlap)[:5]}{'...' if len(overlap) > 5 else ''}"
                )

            basename_to_origin: Dict[str, str] = {}
            for f in fonts:
                bn = Path(f).name
                if bn in basename_to_origin:
                    raise RuntimeError(
                        f"[split:font] Duplicate basename '{bn}' in dataset origins. "
                        f"Cannot use --fixed-split-dir with ambiguous basenames."
                    )
                basename_to_origin[bn] = f

            train_fonts: set = set()
            test_fonts: set = set()
            missing_train = []
            missing_test = []
            for bn in train_basenames:
                if bn in basename_to_origin:
                    train_fonts.add(basename_to_origin[bn])
                else:
                    missing_train.append(bn)
            for bn in test_basenames:
                if bn in basename_to_origin:
                    test_fonts.add(basename_to_origin[bn])
                else:
                    missing_test.append(bn)

            if missing_train:
                print(
                    f"[split:font] WARNING: {len(missing_train)} fonts in train.txt not found in dataset (ignored)"
                )
            if missing_test:
                print(
                    f"[split:font] WARNING: {len(missing_test)} fonts in test.txt not found in dataset (ignored)"
                )

            all_listed = train_fonts | test_fonts
            unlisted = set(fonts) - all_listed
            if unlisted:
                print(
                    f"[split:font] WARNING: {len(unlisted)} fonts in dataset appear in neither "
                    f"train.txt nor test.txt — they will be EXCLUDED from training and evaluation."
                )

            if not train_fonts or not test_fonts:
                raise RuntimeError(
                    f"[split:font] After matching, train_fonts={len(train_fonts)} test_fonts={len(test_fonts)}. "
                    f"Need at least 1 font in each split."
                )

            print(f"[split:font] Loaded fixed split from {fixed_split_dir}")
            print(
                f"[split:font] train.txt: {len(train_basenames)} entries -> {len(train_fonts)} matched"
            )
            print(
                f"[split:font] test.txt:  {len(test_basenames)} entries -> {len(test_fonts)} matched"
            )

        else:

            if len(fonts) == 1:
                raise RuntimeError(
                    "[split:font] Cannot split: only 1 unique font in dataset. "
                    "Need >=2 fonts for a train/test split by font."
                )

            rng.shuffle(fonts)
            cut = int(len(fonts) * args.train_frac)

            if cut >= len(fonts):
                cut = len(fonts) - 1
            if cut <= 0:
                cut = 1

            train_fonts = set(fonts[:cut])
            test_fonts = set(fonts[cut:])

        train_items = [it for it in tuple_items if it.get("origin") in train_fonts]
        test_items = [it for it in tuple_items if it.get("origin") in test_fonts]

        args._train_origins = sorted(train_fonts)
        args._test_origins = sorted(test_fonts)

        def _uniq_fonts(xs):
            return len({it.get("origin") for it in xs})

        def _uniq_uni(xs):
            return len(
                {
                    it.get("unicode_decimal")
                    for it in xs
                    if it.get("unicode_decimal") is not None
                }
            )

        train_fonts_seen = {it.get("origin") for it in train_items}
        test_fonts_seen = {it.get("origin") for it in test_items}

        print(
            f"[split:font] unique_fonts total={len(fonts)} train={len(train_fonts)} test={len(test_fonts)} "
            f"(train_frac={args.train_frac}, seed={args.seed})"
        )
        print(
            f"[split:font] tuples train={len(train_items)} test={len(test_items)} | "
            f"fonts train={_uniq_fonts(train_items)} test={_uniq_fonts(test_items)} | "
            f"unicodes train={_uniq_uni(train_items)} test={_uniq_uni(test_items)}"
        )
        print(
            f"[split:font] overlap_check fonts={len(train_fonts_seen & test_fonts_seen)}"
        )

        from collections import defaultdict

        axis_train_fonts: dict[str, set] = defaultdict(set)
        axis_test_fonts: dict[str, set] = defaultdict(set)
        for it in train_items:
            for a in it.get("axis_names", []):
                axis_train_fonts[a].add(it.get("origin"))
        for it in test_items:
            for a in it.get("axis_names", []):
                axis_test_fonts[a].add(it.get("origin"))
        all_axes = sorted(set(axis_train_fonts) | set(axis_test_fonts))
        print("[split:font] per-axis font counts:")
        for a in all_axes:
            print(
                f"  {a}: train={len(axis_train_fonts[a])} test={len(axis_test_fonts[a])}"
            )

    else:
        raise ValueError(f"Unknown split method: {args.split_method}")

    if args.linepreservationatinference and not hold_in_memory:
        raise ValueError(
            "--linepreservationatinference currently requires --hold-data-in-memory because line-constraint extraction is not implemented for lazy loading."
        )

    if line_loss_enabled:
        print("starting to enrich with line constraints for line loss...")
        t0_enrich = time.time()
        enrich_with_line_constraints(
            train_items,
            hold_in_memory=hold_in_memory,
            axis_spec=axis_spec,
            only_axes=only_axes_set,
            allow_only_single_axis_tuples=args.allow_only_single_axis_tuples,
            max_points_allowed=int(args.max_points_allowed),
            prediction_type=args.prediction_type,
        )
        t1_enrich = time.time()
        n_constraints = sum(len(it.get("line_constraints", [])) for it in train_items)
        print(
            f"[lineloss] enrichment completed in {(t1_enrich - t0_enrich):.3f}s "
            f"for {len(train_items)} train items; constraints={n_constraints}"
        )

    if args.linepreservationatinference:
        print(
            "starting to enrich test items with line constraints for inference-time line preservation..."
        )
        t0_enrich = time.time()
        enrich_with_line_constraints(
            test_items,
            hold_in_memory=hold_in_memory,
            axis_spec=axis_spec,
            only_axes=only_axes_set,
            allow_only_single_axis_tuples=args.allow_only_single_axis_tuples,
            max_points_allowed=int(args.max_points_allowed),
            prediction_type=args.prediction_type,
        )
        t1_enrich = time.time()
        n_constraints = sum(len(it.get("line_constraints", [])) for it in test_items)
        print(
            f"[linepreservationatinference] enrichment completed in {(t1_enrich - t0_enrich):.3f}s "
            f"for {len(test_items)} test items; constraints={n_constraints}"
        )

    if hold_in_memory:
        train_ds = GlyphTupleDataset(train_items)
        test_ds = GlyphTupleDataset(test_items)
    else:
        train_ds = LazyGlyphTupleDataset(
            index_items=train_items,
            axis_spec=axis_spec,
            only_axes=only_axes_set,
            allow_only_single_axis_tuples=args.allow_only_single_axis_tuples,
            max_points_allowed=int(args.max_points_allowed),
            prediction_type=args.prediction_type,
        )
        test_ds = LazyGlyphTupleDataset(
            index_items=test_items,
            axis_spec=axis_spec,
            only_axes=only_axes_set,
            allow_only_single_axis_tuples=args.allow_only_single_axis_tuples,
            max_points_allowed=int(args.max_points_allowed),
            prediction_type=args.prediction_type,
        )
    print(
        f"[data] train_ds={len(train_ds)} test_ds={len(test_ds)} (hold_in_memory={hold_in_memory})"
    )

    max_contour_id_plus_one = max(0, int(max_contour_id_seen) + 1)

    collator = RegressionCollator(
        point_contour_info=int(args.point_contour_info),
        max_contour_id_seen_plus_one=max_contour_id_plus_one,
        device=device if args.pin_to_device else None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.micro_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    max_contour_id_plus_three = max_contour_id_plus_one + 2

    model = GlyphDeltaRegressor(
        num_axes=len(axis_spec.id_to_name),
        max_points=max_points,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        cond_method=args.cond_method,
        axisembed_method=args.axisembed,
        pos_enc=args.pos_enc,
        point_mlp_hidden=args.point_mlp_hidden,
        cond_mlp_hidden=args.cond_mlp_hidden,
        point_contour_info=int(args.point_contour_info),
        num_contour_id_plus_three=max_contour_id_plus_three,
        point_feat_dim=collator.point_feat_dim(),
        polar_regression=bool(args.polar_regression),
        coordnorm=bool(args.coordnorm),
    ).to(device)

    if args.load_checkpoint:
        ckpt_path = _resolve_model_pt(args.load_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--load-checkpoint not found: {ckpt_path}")

        print(f"[load] loading model weights from: {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")
        try:
            if True:

                if any(k.startswith("encoder.") for k in sd.keys()) and not any(
                    k.startswith("cond_transformer_encoder.") for k in sd.keys()
                ):
                    print("BACKWARD COMPAT")
                    sd = {
                        (
                            k.replace("encoder.", "cond_transformer_encoder.", 1)
                            if k.startswith("encoder.")
                            else k
                        ): v
                        for k, v in sd.items()
                    }
            model.load_state_dict(sd, strict=True)
        except RuntimeError as e:

            missing, unexpected = model.load_state_dict(sd, strict=False)
            print("[load] WARNING: strict load failed.")
            print(f"[load] missing keys: {missing}")
            print(f"[load] unexpected keys: {unexpected}")
            raise
        print("[load] done")

    wandb_run = maybe_init_wandb(args)

    def _count_params(m: nn.Module) -> tuple[int, int]:
        total = sum(p.numel() for p in m.parameters())
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return trainable, total

    trainable, total = _count_params(model)
    print("\n[model] architecture:")

    print(
        f"[model] trainable params: {trainable:,} / total params: {total:,} ({trainable/ max(1,total):.2%} trainable)\n"
    )

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_eval_loss: float = float("inf")
    best_step: int = -1
    best_epoch: int = -1

    def maybe_save_best(ev: float, *, step: int, epoch: int) -> None:
        nonlocal best_eval_loss, best_step, best_epoch
        if not args.output_dir:
            return
        if not math.isfinite(ev):
            return
        if ev < best_eval_loss:
            best_eval_loss = float(ev)
            best_step = int(step)
            best_epoch = int(epoch)

            save_checkpoint(args, model, axis_spec, max_points, suffix="best")

            meta = {
                "best_eval_loss": best_eval_loss,
                "best_step": best_step,
                "best_epoch": best_epoch,
            }
            out_dir = Path(args.output_dir)
            (out_dir / "best" / "best_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            print(
                f"[best] updated: eval_loss={best_eval_loss:.6f} at step={best_step} epoch={best_epoch}"
            )

    def eval():
        return evaluate(
            args,
            model,
            test_loader,
            device,
            use_bf16=args.bf16,
            collator=collator,
            print_sample=bool(args.eval_print_sample),
            sample_k=int(args.eval_sample_k),
        )

    if args.only_eval:
        print(
            f"[only-eval] starting evaluation on {len(test_loader.dataset)} test examples ..."
        )
        ev = eval()
        if args.eval_loss == "mse":
            print(
                f"Only eval: eval_{args.eval_loss}_loss={ev:.6f} rmse={math.sqrt(max(0.0, ev)):.6f}"
            )
        else:
            print(f"Only eval: eval_{args.eval_loss}_loss={ev:.6f}")

    else:

        save_checkpoint(args, model, axis_spec, max_points, suffix="before_training")

        step = 0
        model.train()
        num_batches = len(train_loader)

        _last_log_time = time.time()
        _last_log_step = 0

        for epoch in range(args.epochs):
            for batch in train_loader:
                b = move_batch_to_device(batch, device)

                max_points_seen_in_batch = int((~b.pad_mask).sum(dim=1).max().item())

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=(args.bf16 and device.type == "cuda"),
                ):

                    pred = model(
                        b.points,
                        b.contour_tok_id,
                        b.axis_id,
                        b.axis_value,
                        b.pad_mask,
                        b.axis_mask,
                        combinatorial_max_order=args.combinatorial_max_order,
                    )

                    per = compute_loss(
                        pred, b.deltas, args.train_loss, args.huber_delta
                    )
                    per = per.masked_fill(b.pad_mask, 0.0)
                    denom_num_valid_points_in_batch = (~b.pad_mask).sum().clamp(min=1)

                    if not line_loss_enabled:
                        loss = per.sum() / denom_num_valid_points_in_batch
                    else:
                        base_loss = per.sum() / denom_num_valid_points_in_batch
                        line_loss = (
                            compute_line_preservation_loss(pred, b)
                            / denom_num_valid_points_in_batch
                        )
                        loss = base_loss + float(line_loss_alpha) * line_loss

                    loss = loss / max(1, args.grad_accum_steps)

                loss.backward()

                if (step + 1) % args.grad_accum_steps == 0:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.grad_clip
                        )
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                if step % args.logging_steps == 0:
                    _now = time.time()
                    _steps_since = max(1, step - _last_log_step)
                    _sec_per_step = (_now - _last_log_time) / _steps_since
                    _last_log_time = _now
                    _last_log_step = step
                    if line_loss_enabled:
                        print(
                            f"[step {step}/{num_batches}, e {epoch}] "
                            f"train total_loss={loss.detach().float().item():.6f} "
                            f"base_{args.train_loss}={base_loss.detach().float().item():.6f} "
                            f"line_loss={line_loss.detach().float().item():.6f} "
                            f"batch_max_points={max_points_seen_in_batch} sec/step={_sec_per_step:.3f}"
                        )
                    else:
                        print(
                            f"[step {step}/{num_batches}, e {epoch}] train {args.train_loss}_loss="
                            f"{loss.detach().float().item():.6f} "
                            f"batch_max_points={max_points_seen_in_batch} sec/step={_sec_per_step:.3f}"
                        )

                if args.eval_steps > 0 and step > 0 and step % args.eval_steps == 0:
                    ev = eval()
                    print(f"[step {step}] eval_{args.eval_loss}_loss={ev:.6f}")
                    maybe_wandb_log(
                        args, {f"eval/{args.eval_loss}_loss": float(ev)}, step=step
                    )
                    maybe_save_best(ev, step=step, epoch=epoch)

                step += 1

            ev = eval()

            print(f"[epoch {epoch}] eval_{args.eval_loss}_loss={ev:.6f}")
            maybe_wandb_log(
                args,
                {f"eval/{args.eval_loss}_loss": float(ev), "epoch": int(epoch)},
                step=step,
            )
            maybe_save_best(ev, step=step, epoch=epoch)

            if args.save_each_epoch:
                save_checkpoint(
                    args, model, axis_spec, max_points, suffix=f"epoch_{epoch}"
                )

        save_checkpoint(args, model, axis_spec, max_points, suffix="final")

    if wandb_run is not None:
        import wandb

        wandb.finish()


def save_checkpoint(
    args, model: nn.Module, axis_spec: AxisSpec, max_points: int, suffix: str
) -> None:
    if not args.output_dir:
        print("[not saving model]")
        return
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = out_dir / suffix
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), ckpt_dir / "model.pt")
    cfg = dict(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        cond_method=args.cond_method,
        axisembed_method=args.axisembed,
        combinatorial_max_order=args.combinatorial_max_order,
        pos_enc=args.pos_enc,
        point_mlp_hidden=args.point_mlp_hidden,
        cond_mlp_hidden=args.cond_mlp_hidden,
        max_points=max_points,
        axes=axis_spec.id_to_name,
        point_contour_info=int(args.point_contour_info),
        prediction_type=args.prediction_type,
        split_method=args.split_method,
        lineloss=args.lineloss,
        polar_regression=bool(args.polar_regression),
        coordnorm=bool(args.coordnorm),
    )
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    if args.split_method == "font":
        train_list = args._train_origins
        test_list = args._test_origins
        if train_list is not None and test_list is not None:
            train_names = [Path(str(x)).name for x in train_list]
            test_names = [Path(str(x)).name for x in test_list]
            (ckpt_dir / "train.txt").write_text(
                "\n".join(train_names) + "\n", encoding="utf-8"
            )
            (ckpt_dir / "test.txt").write_text(
                "\n".join(test_names) + "\n", encoding="utf-8"
            )

    if args.split_method == "unicode":
        train_unis = args._train_unicodes
        test_unis = args._test_unicodes
        if train_unis is not None and test_unis is not None:
            (ckpt_dir / "train_unicodes.txt").write_text(
                "\n".join(str(u) for u in train_unis) + "\n", encoding="utf-8"
            )
            (ckpt_dir / "test_unicodes.txt").write_text(
                "\n".join(str(u) for u in test_unis) + "\n", encoding="utf-8"
            )

    print(f"[save] {ckpt_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=False)

    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--train-frac", type=float, default=0.85)

    ap.add_argument(
        "--only-axes",
        type=str,
        default="",
        help=(
            'Comma-separated list of axes to keep (e.g. "wght,wdth"). '
            "When set, a tuple is kept only if ALL its <coord> axes are within this set."
        ),
    )
    ap.add_argument(
        "--allow-only-single-axis-tuples",
        action="store_true",
        help=(
            "When set, keep only tuples with exactly one <coord> entry (and that axis must be in --only-axes if provided)."
        ),
    )

    ap.add_argument(
        "--limit-examples",
        type=int,
        default=0,
        help=(
            "Cap the total number of training items (tuples) loaded from the dataset. "
            "0 = no limit (default). Useful for quick debugging runs with a small subset. "
            "Note: this counts tuples, not glyphs — a single glyph may contribute "
            "multiple tuples (one per axis variation)."
        ),
    )
    ap.add_argument(
        "--max-points-allowed",
        type=int,
        default=500,
        help="Skip glyphs whose base contour points exceed this number (phantom points not included).",
    )

    ap.add_argument(
        "--point-contour-info",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help=(
            "0=no contour info. Model is not aware what point belongs to which contour; "
            "1=contour-id embedding; "
            "2=is_new_contour scalar; "
            "3=pos_in_contour scalar, normalized to [0,1]"
        ),
    )

    ap.add_argument("--d-model", "--d_model", type=int, default=256)

    ap.add_argument("--num-layers", "--num_layers", type=int, default=6)
    ap.add_argument("--num-heads", "--num_heads", type=int, default=8)
    ap.add_argument("--ff-mult", "--ff_mult", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pos-enc", choices=["sinusoidal", "none"], default="sinusoidal")

    ap.add_argument(
        "--cond-method",
        "--cond_method",
        choices=["token", "film", "adaln", "adaln2", "cross"],
        default="adaln",
    )

    ap.add_argument(
        "--axisembed",
        choices=["addall", "mult", "concat", "combinatorial"],
        default="combinatorial",
        help="Axis embedding strategy: 'addall', 'mult', 'concat', or 'combinatorial' (default: 'addall'). "
        "'combinatorial' creates 3^num_axes-1 embeddings for all axis combinations with pos/neg variants.",
    )
    ap.add_argument(
        "--combinatorial-max-order",
        type=int,
        default=1,
        help="If set, limit combinatorial embedding to interactions of at most this order "
        "(1=single-axis only, 2=up to pairwise, etc.). For ablation studies.",
    )

    ap.add_argument("--point-mlp-hidden", type=int, default=128)
    ap.add_argument("--cond-mlp-hidden", type=int, default=128)
    ap.add_argument(
        "--polar-regression",
        "--polar_regression",
        action="store_true",
        help=(
            "If set, predict (r, cos_theta, sin_theta) at the head, normalize (cos,sin), "
            "convert to (dx,dy), and train/evaluate as usual in Cartesian space."
        ),
    )
    ap.add_argument(
        "--coordnorm",
        action="store_true",
        help=(
            "If set, apply per-glyph bbox coordinate normalization in model forward on regular points only, "
            "with matching output denormalization (CoordNorm/CoordDenorm)."
        ),
    )

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--micro-batch-size", "--micro_batch_size", type=int, default=32)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no-bf16", dest="bf16", action="store_false")

    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument(
        "--lineloss",
        "--loss1",
        dest="lineloss",
        type=float,
        default=None,
        help=(
            "Weight alpha for straight-line preservation loss added during training only. "
            "None (default) disables it. Example: --lineloss 0.5"
        ),
    )

    ap.add_argument(
        "--train-loss",
        type=str,
        choices=["huber", "mse"],
        required=True,
        help="Loss function for training: 'huber' (Huber/smooth-L1) or 'mse' (mean squared error).",
    )
    ap.add_argument(
        "--linepreservationatinference",
        action="store_true",
        help=(
            "If set, evaluation/--only-eval additionally projects predicted points that belong to detected straight-line "
            "groups onto a total-least-squares fitted line. This does not change training loss or model weights."
        ),
    )
    ap.add_argument(
        "--eval-loss",
        type=str,
        choices=["huber", "mse"],
        required=True,
        help="Loss/metric for evaluation: 'huber' or 'mse'. Reported in logs and wandb as eval/<name>_loss.",
    )

    ap.add_argument("--logging-steps", type=int, default=50)
    ap.add_argument("--eval-steps", type=int, default=1000)

    ap.add_argument(
        "--no-eval-print-sample",
        dest="eval_print_sample",
        action="store_false",
        default=True,
        help="Disable printing a debug sample during evaluate()",
    )
    ap.add_argument(
        "--eval-sample-k",
        type=int,
        default=500,
        help="How many points to print from the sampled glyph (includes phantom points).",
    )

    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--pin-to-device",
        action="store_true",
        help="Move tensors to GPU inside collator.",
    )

    ap.add_argument(
        "--dont-hold-data-in-memory",
        dest="hold_data_in_memory",
        action="store_false",
        default=True,
        help=(
            "When set, do NOT load all training data into RAM. Instead, only a lightweight "
            "index is loaded and glyph records are read from disk on demand, drastically "
            "reducing memory usage. By default, all data is loaded into RAM upfront."
        ),
    )

    ap.add_argument(
        "--wandb",
        action="store_true",
        help="If set, log eval loss to Weights & Biases.",
    )
    ap.add_argument("--wandb-project", type=str, default="autovarfont")
    ap.add_argument("--wandb-run-name", type=str, default=None)

    ap.add_argument(
        "--save-each-epoch", action="store_true", help="Also save after each epoch."
    )

    ap.add_argument(
        "--load-checkpoint",
        type=Path,
        default=None,
        help="Path to a checkpoint dir (contains model.pt) or directly to model.pt. If set, load weights before training.",
    )
    ap.add_argument(
        "--only-eval", action="store_true", help="If set, doing only eval, no training."
    )
    ap.add_argument(
        "--nullbaseline",
        action="store_true",
        help="Only for --only-eval: skip model inference and predict zero deltas for evaluation.",
    )
    ap.add_argument(
        "--only-eval-no-high-order",
        dest="only_eval_no_high_order",
        action="store_true",
        help=(
            "Only works with --only-eval. Ablation: for examples with multiple active axes, "
            "run the model once per axis (masking out the others) and SUM the predictions "
            "instead of feeding all axes together."
        ),
    )

    ap.add_argument(
        "--split-method",
        choices=["item", "glyph", "glyphname", "unicode", "font"],
        required=True,
        help=(
            "How to split train/test: "
            "'item' = [not recommended] random over <tuple>s (current); It means that for some glyph, one tuple can be found in train and another in test; "
            "'glyph' = split by glyph_id (origin, glyph_name); the same glyph cannot appear in both train and test, but glyphname 'A' in font X can go to train and 'A' in Y can go to test. "
            "'glyphname' = split by glyph_name; the same glyph name cannot appear in both train and test; "
            "'unicode' = split by unicode_decimal; the same unicode cannot appear in both train and test; "
            "'font' = split by font file so each font is disjoint."
        ),
    )
    ap.add_argument(
        "--fixed-split-dir",
        type=Path,
        default=None,
        help=(
            "Path to a directory containing pre-defined split files. "
            "With --split-method font: expects train.txt and test.txt (one font filename per line). "
            "With --split-method unicode: expects train_unicodes.txt and test_unicodes.txt (one unicode decimal per line). "
            "When provided, the train/test split is loaded from these files instead of being randomly generated, "
            "ensuring reproducibility across teams."
        ),
    )

    ap.add_argument(
        "--prediction-type",
        "--prediction_type",
        type=str,
        choices=["tuple", "total", "residual"],
        default="total",
        help=(
            "Prediction type: 'tuple' (default) = model predicts single tuple delta from glyph original outline; "
            "'total' = model predicts accumulated delta from all tuples in glyph at given axis configuration; glyph input outline is original; the axes coordinates are taken from a single tuple, but the label is taken from all tuples w.r.t to the axes configiration as specified by this tuple."
            "'residual' = accumulate deltas from other tuples (subset axes only), add to glyph oncurve as input, and predict residual delta for a single tuple."
        ),
    )

    args = ap.parse_args()

    if args.only_eval_no_high_order and not args.only_eval:
        raise ValueError("--only-eval-no-high-order requires --only-eval")
    if args.nullbaseline and not args.only_eval:
        raise ValueError("--nullbaseline requires --only-eval")
    if args.cond_method == "cross" and args.axisembed != "combinatorial":
        raise ValueError("--cond-method cross requires --axisembed combinatorial")

    if not is_cuda():
        print("CUDA is not available. This script is intended to run on GPU.")
        return

    set_all_seeds(int(args.seed))
    train(args)


if __name__ == "__main__":
    main()
