from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


TTGLYPH_RE = re.compile(r"<TTGlyph\b[^>]*>(.*?)</TTGlyph>", re.DOTALL)
GVAR_RE = re.compile(r"<glyphVariations\b[^>]*>(.*?)</glyphVariations>", re.DOTALL)
GLYPH_RECORD_RE = re.compile(r"(<GlyphRecord>.*?</GlyphRecord>)", re.DOTALL)
TTGLYPH_FULL_RE = re.compile(r"<TTGlyph\b([^>]*)>(.*?)</TTGlyph>", re.DOTALL)


UNICODE_DEC_RE = re.compile(r"<unicode\b[^>]*\bdec=\"(\d+)\"", re.DOTALL)


def iter_all_pairs_with_glyph_name(
    data_dir: Path,
) -> Iterable[Tuple[str, str, str, str, Optional[int]]]:

    paths = list(data_dir.rglob("*.xml")) + list(data_dir.rglob("*.xml.gz"))

    seen_names = set()
    for font_xml_path in paths:
        if font_xml_path.name in seen_names:
            print(f"[ERROR] Duplicate font name found: {font_xml_path.name}")
            exit(1)
        seen_names.add(font_xml_path.name)

    for font_xml_path in paths:
        try:
            txt = read_text_maybe_gz(font_xml_path)
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
                yield inner, gvar_inner, str(font_xml_path), glyph_name, unicode_decimal
        except Exception as e:
            print(f"[skip] {font_xml_path}: {e}")


def read_text_maybe_gz(path: Path) -> str:
    if path.suffix == ".gz":
        raise NotImplementedError("gzip should not exist")

    return path.read_text(encoding="utf-8")


def extract_pairs_from_font_xml(full_xml_text: str) -> List[Tuple[str, str]]:

    pairs: List[Tuple[str, str]] = []
    for rec_m in GLYPH_RECORD_RE.finditer(full_xml_text):
        rec = rec_m.group(1)
        m1 = TTGLYPH_RE.search(rec)
        m2 = GVAR_RE.search(rec)
        if not m1 or not m2:
            continue
        pairs.append((m1.group(1).strip(), m2.group(1).strip()))
    return pairs


def iter_all_pairs(data_dir: Path) -> Iterable[Tuple[str, str, str]]:

    paths = list(data_dir.rglob("*.xml"))
    for path in paths:
        try:
            xml_txt = read_text_maybe_gz(path)
            for input_txt, label_txt in extract_pairs_from_font_xml(xml_txt):
                yield input_txt, label_txt, str(path)
        except Exception as e:
            print(f"[skip] {path}: {e}")


def compress_training_string(s: str) -> str:

    s = s.replace("<axes>", "<a>")
    s = s.replace("</axes>", "</a>")
    s = s.replace("<coord axis", "<a")
    s = s.replace("value", "v")
    s = s.replace("<contour>", "<c>")
    s = s.replace("</contour>", "</c>")
    s = s.replace("<pt x", "<x")
    s = s.replace("on", "o")
    s = s.replace("=", "")
    s = s.replace("<delta pt", "<d")
    s = s.replace("<tuple>", "<t>")
    s = s.replace("</tuple>", "</t>")
    s = "".join(s.split())
    return s


def make_prompt(src_ttglyph_inner: str) -> str:

    src_ttglyph_inner = compress_training_string(src_ttglyph_inner)
    return (
        "TASK: Given a TrueType glyph outline, predict its glyph variations (gvar XML).\n"
        "INPUT:\n"
        f"{src_ttglyph_inner}\n"
        "OUTPUT:\n"
    )


AXES_TAG_RE = re.compile(r"<axes>(.*?)</axes>", re.DOTALL)


CONTOUR_RE = re.compile(r"<contour>(.*?)</contour>", re.DOTALL)
PT_RE = re.compile(
    r"<pt\b[^>]*\bx=\"([-\d\.]+)\"\s+y=\"([-\d\.]+)\"\s+on=\"([01])\"[^>]*/?>"
)


TUPLE_RE = re.compile(r"<tuple>(.*?)</tuple>", re.DOTALL)
COORD_RE = re.compile(
    r'<coord\b[^>]*\baxis="([^"]+)"'
    r'(?:[^>]*\bmin="([\-\d\.]+)")?'
    r'[^>]*\bvalue="([\-\d\.]+)"'
    r'(?:[^>]*\bmax="([\-\d\.]+)")?'
    r"[^>]*/?>"
)
DELTA_RE = re.compile(
    r"<delta\b[^>]*\bpt=\"(\d+)\"\s+x=\"([-\d\.]+)\"\s+y=\"([-\d\.]+)\"[^>]*/?>"
)


@dataclass
class ParsedGlyph:
    points_xy: List[Tuple[float, float]]
    oncurve: List[int]
    contour_id: List[int]
    axes_declared: List[str]


@dataclass
class ParsedTuple:
    coords: Dict[str, Dict[str, float]]
    deltas: List[Tuple[int, float, float]]


def parse_ttglyph_inner(t: str) -> ParsedGlyph:
    axes_declared: List[str] = []
    m = AXES_TAG_RE.search(t)
    if m:
        axes_declared = [a.strip() for a in m.group(1).split(",") if a.strip()]

    points_xy: List[Tuple[float, float]] = []
    oncurve: List[int] = []
    contour_id: List[int] = []

    cid = 0
    for cm in CONTOUR_RE.finditer(t):
        cbody = cm.group(1)
        found_any = False
        for pm in PT_RE.finditer(cbody):
            found_any = True
            x = float(pm.group(1))
            y = float(pm.group(2))
            o = int(pm.group(3))
            points_xy.append((x, y))
            oncurve.append(o)
            contour_id.append(cid)
        if found_any:
            cid += 1

    return ParsedGlyph(
        points_xy=points_xy,
        oncurve=oncurve,
        contour_id=contour_id,
        axes_declared=axes_declared,
    )


def parse_gvar_inner(g: str) -> List[ParsedTuple]:
    tuples: List[ParsedTuple] = []
    for tm in TUPLE_RE.finditer(g):
        tbody = tm.group(1)
        coords: Dict[str, Dict[str, float]] = {}
        for cm in COORD_RE.finditer(tbody):
            axis_name = cm.group(1)
            min_str = cm.group(2)
            value_str = cm.group(3)
            max_str = cm.group(4)

            coord_dict = {"value": float(value_str)}
            if min_str is not None:
                coord_dict["min"] = float(min_str)
            if max_str is not None:
                coord_dict["max"] = float(max_str)

            coords[axis_name] = coord_dict

        deltas: List[Tuple[int, float, float]] = []
        for dm in DELTA_RE.finditer(tbody):
            pt = int(dm.group(1))
            dx = float(dm.group(2))
            dy = float(dm.group(3))
            deltas.append((pt, dx, dy))

        tuples.append(ParsedTuple(coords=coords, deltas=deltas))
    return tuples
