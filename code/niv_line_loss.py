from __future__ import annotations


import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import tqdm

from shared import GLYPH_RECORD_RE, TTGLYPH_FULL_RE, parse_ttglyph_inner


def _extract_line_constraints_from_points(
    points_xy: List[Tuple[float, float]],
    oncurve: List[float],
    contour_id: List[int],
    *,
    min_points_on_line: int = 3,
    min_pair_len: float = 1e-9,
    line_dist_tol: float = 0.5,
    dir_tol: float = 1e-5,
    touch_tol: float = 1e-9,
) -> List[Dict]:

    assert (
        int(min_points_on_line) >= 3
    ), "line constraints must be built from segments with at least 3 points"
    if not points_xy or not oncurve or not contour_id:
        return []
    if len(points_xy) != len(oncurve) or len(points_xy) != len(contour_id):
        return []

    n = len(points_xy)
    if n < min_points_on_line:
        return []

    def _canonical_line_from_points(i: int, j: int):
        xi, yi = points_xy[i]
        xj, yj = points_xy[j]
        dx = float(xj) - float(xi)
        dy = float(yj) - float(yi)
        norm = math.hypot(dx, dy)
        if norm <= min_pair_len:
            return None

        ux = dx / norm
        uy = dy / norm
        if (ux < 0.0) or (abs(ux) <= 1e-12 and uy < 0.0):
            ux = -ux
            uy = -uy
        nx = -uy
        ny = ux
        c = nx * float(xi) + ny * float(yi)
        return ux, uy, nx, ny, c

    def _same_supporting_line(a, b) -> bool:
        ux1, uy1, _, _, c1 = a
        ux2, uy2, _, _, c2 = b
        cross = abs(ux1 * uy2 - uy1 * ux2)
        return cross <= dir_tol and abs(c1 - c2) <= line_dist_tol

    def _run_from_point_indices(idxs: List[int]):
        if len(idxs) < 2:
            return None
        lr = _canonical_line_from_points(idxs[0], idxs[-1])
        if lr is None:

            for k in range(len(idxs) - 1):
                lr = _canonical_line_from_points(idxs[k], idxs[k + 1])
                if lr is not None:
                    break
        if lr is None:
            return None
        ux, uy, _, _, _ = lr
        tvals = [
            ux * float(points_xy[p][0]) + uy * float(points_xy[p][1]) for p in idxs
        ]
        return {
            "points": list(idxs),
            "line": lr,
            "t_min": min(tvals),
            "t_max": max(tvals),
        }

    contours: Dict[int, List[int]] = {}
    for i, cid in enumerate(contour_id):
        contours.setdefault(int(cid), []).append(i)

    runs: List[Dict[str, Any]] = []
    for idxs in contours.values():
        m = len(idxs)
        if m < 2:
            continue

        edge_valid: List[bool] = []
        edge_line: List[Optional[Tuple[float, float, float, float, float]]] = []
        for k in range(m):
            i = idxs[k]
            j = idxs[(k + 1) % m]
            if float(oncurve[i]) <= 0.5 or float(oncurve[j]) <= 0.5:
                edge_valid.append(False)
                edge_line.append(None)
                continue
            lr = _canonical_line_from_points(i, j)
            if lr is None:
                edge_valid.append(False)
                edge_line.append(None)
                continue
            edge_valid.append(True)
            edge_line.append(lr)

        if not any(edge_valid):
            continue

        start = 0
        for k in range(m):
            if not edge_valid[k]:
                start = (k + 1) % m
                break
        order = [(start + k) % m for k in range(m)]

        k = 0
        while k < m:
            eidx = order[k]
            if not edge_valid[eidx]:
                k += 1
                continue

            current_line = edge_line[eidx]
            run_pts = [idxs[eidx], idxs[(eidx + 1) % m]]
            k += 1
            while k < m:
                neidx = order[k]
                if not edge_valid[neidx]:
                    break
                nline = edge_line[neidx]
                if (
                    current_line is None
                    or nline is None
                    or not _same_supporting_line(current_line, nline)
                ):
                    break
                run_pts.append(idxs[(neidx + 1) % m])
                k += 1

            dedup_pts: List[int] = []
            for p in run_pts:
                if not dedup_pts or dedup_pts[-1] != p:
                    dedup_pts.append(p)

            run = _run_from_point_indices(dedup_pts)
            if run is not None:
                runs.append(run)

    if not runs:
        return []

    parent = list(range(len(runs)))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra = _find(a)
        rb = _find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(runs)):
        li = runs[i]["line"]
        l1 = float(runs[i]["t_min"])
        r1 = float(runs[i]["t_max"])
        for j in range(i + 1, len(runs)):
            lj = runs[j]["line"]
            if not _same_supporting_line(li, lj):
                continue
            l2 = float(runs[j]["t_min"])
            r2 = float(runs[j]["t_max"])
            if max(l1, l2) <= min(r1, r2) + touch_tol:
                _union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(len(runs)):
        root = _find(i)
        groups.setdefault(root, []).append(i)

    constraints: List[Dict] = []
    seen_point_sets: Set[Tuple[int, ...]] = set()
    for g in groups.values():
        rep = runs[g[0]]
        ux, uy, _, _, _ = rep["line"]

        point_t: Dict[int, float] = {}
        for ridx in g:
            for p in runs[ridx]["points"]:
                x, y = points_xy[p]
                point_t[int(p)] = ux * float(x) + uy * float(y)

        ordered = sorted(point_t.items(), key=lambda kv: (kv[1], kv[0]))
        ordered_idx = [int(p) for p, _ in ordered]
        if len(ordered_idx) < min_points_on_line:
            continue

        pset = tuple(ordered_idx)
        if pset in seen_point_sets:
            continue
        seen_point_sets.add(pset)

        constraints.append(
            {
                "anchor_i": int(ordered_idx[0]),
                "anchor_j": int(ordered_idx[-1]),
                "inner_idx": [int(k) for k in ordered_idx[1:-1]],
            }
        )
        assert 2 + len(constraints[-1]["inner_idx"]) >= 3, (
            "invalid line constraint: segment must have at least 3 total points "
            f"(got {2 + len(constraints[-1]['inner_idx'])})"
        )

    return constraints


def enrich_with_line_constraints(
    items: List[Dict],
    *,
    hold_in_memory: bool,
    **_: Any,
) -> None:

    if not items:
        return
    if not hold_in_memory:
        raise TypeError(
            "Line constraint extraction without hold_in_memory is not implemented yet."
        )

    total_items = len(items)
    hist: Counter[int] = Counter()
    last_seen_by_bucket: Dict[int, Tuple[str, str]] = {}

    report_every = 500000

    def _print_hist(prefix: str, processed: int) -> None:
        tqdm.tqdm.write(
            f"[line-constraints-stats] {prefix} processed={processed}/{total_items}"
        )
        if not hist:
            tqdm.tqdm.write("[line-constraints-stats]   constraints=0 glyphs=0")
            return
        for k in sorted(hist.keys()):
            font_name, glyph_name = last_seen_by_bucket.get(k, ("?", "?"))
            tqdm.tqdm.write(
                f"[line-constraints-stats]   constraints={k} glyphs={int(hist[k])} "
                f"last_font={font_name} last_glyph={glyph_name}"
            )

    for idx, it in enumerate(
        tqdm.tqdm(
            items,
            desc="Enriching train items with line constraints (out of items/tuples)",
        ),
        start=1,
    ):
        pts = it.get("points_xy", [])
        oncurve = it.get("oncurve", [])
        contour_id = it.get("contour_id", [])
        constraints = _extract_line_constraints_from_points(pts, oncurve, contour_id)
        it["line_constraints"] = constraints

        n_constraints = len(constraints)
        hist[n_constraints] += 1
        last_seen_by_bucket[n_constraints] = (
            str(it.get("origin", "?")),
            str(it.get("glyph_name", "?")),
        )

        if idx % report_every == 0:
            _print_hist(prefix="partial", processed=idx)

    if total_items % report_every != 0:
        _print_hist(prefix="final", processed=total_items)


def compute_line_preservation_loss(pred: torch.Tensor, b: Any) -> torch.Tensor:

    import torch

    device = pred.device
    dtype = pred.dtype
    total = torch.zeros((), device=device, dtype=dtype)
    eps = torch.tensor(1e-12, device=device, dtype=dtype)

    for bi, meta in enumerate(b.meta):
        constraints = meta.get("line_constraints", [])
        if not constraints:
            continue

        out_xy = b.points[bi, :, :2] + pred[bi]
        for c in constraints:
            i = int(c["anchor_i"])
            j = int(c["anchor_j"])
            inner = c.get("inner_idx", [])
            assert 2 + len(inner) >= 3, (
                "invalid line constraint at training time: segment must have at least 3 total points "
                f"(got {2 + len(inner)}), anchor_i={i}, anchor_j={j}"
            )
            seg_idx = [i] + [int(k) for k in inner] + [j]
            seg = torch.tensor(seg_idx, device=device, dtype=torch.long)
            n_seg = int(seg.numel())

            for t in range(n_seg):
                p_idx = seg[t]
                others = seg[torch.arange(n_seg, device=device) != t]
                if int(others.numel()) < 2:
                    continue

                max_tries = 5
                for _ in range(max_tries):
                    perm = torch.randperm(int(others.numel()), device=device)
                    a_idx = others[perm[0]]
                    b_idx = others[perm[1]]

                    a = out_xy[a_idx]
                    d = out_xy[b_idx] - a
                    denom2 = (d * d).sum()
                    if bool((denom2 > eps).item()):
                        p = out_xy[p_idx]
                        rel = p - a
                        cross = d[0] * rel[1] - d[1] * rel[0]
                        total = total + (cross * cross / denom2)
                        break

    return total


def fit_line_total_least_squares(
    points_xy: List[Tuple[float, float]],
) -> Optional[Dict[str, Tuple[float, float]]]:

    if len(points_xy) < 2:
        return None

    xs = [float(x) for x, _ in points_xy]
    ys = [float(y) for _, y in points_xy]
    mx = sum(xs) / float(len(xs))
    my = sum(ys) / float(len(ys))

    sxx = 0.0
    sxy = 0.0
    syy = 0.0
    for x, y in points_xy:
        dx = float(x) - mx
        dy = float(y) - my
        sxx += dx * dx
        sxy += dx * dy
        syy += dy * dy

    trace = sxx + syy
    det_term = (sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy
    lam = 0.5 * (trace + math.sqrt(max(0.0, det_term)))

    vx = sxy
    vy = lam - sxx
    norm = math.hypot(vx, vy)
    if norm <= 1e-12:
        if sxx >= syy:
            vx, vy = 1.0, 0.0
        else:
            vx, vy = 0.0, 1.0
        norm = 1.0

    ux = vx / norm
    uy = vy / norm
    if (ux < 0.0) or (abs(ux) <= 1e-12 and uy < 0.0):
        ux = -ux
        uy = -uy

    return {
        "center": (mx, my),
        "direction": (ux, uy),
    }


def project_points_onto_fitted_line(
    points_xy: List[Tuple[float, float]],
    *,
    move_mask: Optional[List[bool]] = None,
) -> List[Tuple[float, float]]:

    fit = fit_line_total_least_squares(points_xy)
    if fit is None:
        return list(points_xy)

    cx, cy = fit["center"]
    ux, uy = fit["direction"]
    out: List[Tuple[float, float]] = []
    for i, (x, y) in enumerate(points_xy):
        if move_mask is not None and not bool(move_mask[i]):
            out.append((float(x), float(y)))
            continue
        dx = float(x) - cx
        dy = float(y) - cy
        t = dx * ux + dy * uy
        out.append((cx + t * ux, cy + t * uy))
    return out


def apply_line_preservation_inference(
    points_xy: List[Tuple[float, float]],
    line_constraints: List[Dict],
    *,
    move_anchors: bool = True,
) -> List[Tuple[float, float]]:

    if not line_constraints:
        return list(points_xy)

    corrected = [(float(x), float(y)) for x, y in points_xy]
    for c in line_constraints:
        i = int(c["anchor_i"])
        j = int(c["anchor_j"])
        inner = [int(k) for k in c.get("inner_idx", [])]
        seg_idx = [i] + inner + [j]
        if len(seg_idx) < 3:
            continue
        seg_pts = [corrected[k] for k in seg_idx]
        move_mask = [bool(move_anchors)] + [True] * len(inner) + [bool(move_anchors)]
        seg_corr = project_points_onto_fitted_line(seg_pts, move_mask=move_mask)
        for p_idx, xy_new in zip(seg_idx, seg_corr):
            corrected[p_idx] = xy_new
    return corrected


def _extract_points_for_glyph(
    xml_path: Path, glyph_name: str
) -> Tuple[List[Tuple[float, float]], List[int], List[int]]:
    txt = xml_path.read_text(encoding="utf-8")
    for rec_m in GLYPH_RECORD_RE.finditer(txt):
        rec = rec_m.group(1)
        m = TTGLYPH_FULL_RE.search(rec)
        if not m:
            continue
        attrs = m.group(1) or ""
        inner = (m.group(2) or "").strip()
        if f'name="{glyph_name}"' not in attrs:
            continue

        g = parse_ttglyph_inner(inner)
        return list(g.points_xy), list(g.oncurve), list(g.contour_id)

    raise ValueError(f"Glyph '{glyph_name}' not found in {xml_path}")


def _run_tester(xml_path: Path, glyph_name: str) -> None:
    points_xy, oncurve, contour_id = _extract_points_for_glyph(xml_path, glyph_name)
    constraints = _extract_line_constraints_from_points(points_xy, oncurve, contour_id)

    print(f"xml_path={xml_path}")
    print(f"glyph_name={glyph_name}")
    print(
        f"num_points={len(points_xy)} oncurve_points={sum(1 for x in oncurve if int(x) == 1)}"
    )
    print(f"num_line_constraints={len(constraints)}")
    print(json.dumps(constraints, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Line-constraint extraction tester for one glyph in one XML font file."
    )
    ap.add_argument(
        "--xml-path", type=Path, required=True, help="Path to font XML file."
    )
    ap.add_argument(
        "--glyph-name", type=str, required=True, help='Glyph name, e.g. "A".'
    )
    args = ap.parse_args()
    _run_tester(args.xml_path, args.glyph_name)


if __name__ == "__main__":
    main()
