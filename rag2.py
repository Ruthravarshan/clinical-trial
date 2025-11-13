# Python
import os
import time
import fitz
import google.generativeai as genai
import faiss
import numpy as np
import camelot
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any

# -----------------------------
# Config
# -----------------------------
API_KEY = os.getenv("GEMINI_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)
EMBED_MODEL = "text-embedding-004"

# -----------------------------
# Utilities
# -----------------------------
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna("").astype(str)
    df = df.apply(lambda col: col.str.strip())
    mask = ~df.apply(lambda r: "".join(r.astype(str)).strip() == "", axis=1)
    return df[mask]

def _content_density(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    total = df.shape[0] * df.shape[1]
    nonempty = (df.apply(lambda col: col != "")).values.sum()
    return float(nonempty) / float(total) if total else 0.0

def _page_size(pdf_path: str, page_num: int) -> Tuple[float, float]:
    with fitz.open(pdf_path) as doc:
        rect = doc[page_num - 1].rect
        return float(rect.width), float(rect.height)

def _camelot_bbox(tbl) -> Optional[Tuple[float, float, float, float]]:
    bbox = getattr(tbl, "_bbox", None)
    if bbox and len(bbox) == 4:
        return tuple(map(float, bbox))
    return None

def _bbox_to_areas(bbox: Tuple[float, float, float, float], page_h: float) -> Tuple[str, str]:
    x1, y1, x2, y2 = bbox
    area_bl = f"{x1},{y1},{x2},{y2}"                 # bottom-left origin
    y1_t = page_h - y1
    y2_t = page_h - y2
    area_tl = f"{x1},{y2_t},{x2},{y1_t}"             # top-left origin variant
    return area_bl, area_tl

def _expand_area(area_str: str, pad: float, page_w: float, page_h: float) -> str:
    x1, y1, x2, y2 = map(float, area_str.split(","))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(page_w, x2 + pad)
    y2 = min(page_h, y2 + pad)
    return f"{x1},{y1},{x2},{y2}"

def _read_tables(pdf_path: str, page: int, flavor: str, table_areas: Optional[List[str]] = None,
                 strip_text: str = "n", process_background: bool = True,
                 line_scale: int = 40, row_tol: int = 12):
    try:
        kwargs: Dict[str, Any] = dict(pages=str(page), flavor=flavor, strip_text=strip_text)
        if table_areas:
            kwargs["table_areas"] = table_areas
        if flavor == "lattice":
            kwargs["process_background"] = process_background
            kwargs["line_scale"] = line_scale
        else:
            kwargs["row_tol"] = row_tol
        return camelot.read_pdf(pdf_path, **kwargs)
    except Exception:
        return None

def _collect_candidates(pdf_path: str, page: int, prefer_stream: bool) -> List[Dict[str, Any]]:
    flavors_order = ["stream", "lattice"] if prefer_stream else ["lattice", "stream"]
    page_w, page_h = _page_size(pdf_path, page)
    candidates: List[Dict[str, Any]] = []
    for flv in flavors_order:
        t = _read_tables(pdf_path, page, flavor=flv)
        if not t or t.n == 0:
            continue
        for tbl in t:
            df = _clean_df(tbl.df)
            if df is None or df.empty:
                continue
            bbox = _camelot_bbox(tbl)
            ar = 0.0
            if bbox:
                x1, y1, x2, y2 = bbox
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                page_area = page_w * page_h
                ar = (w * h) / page_area if page_area > 0 else 0.0
            candidates.append(dict(
                flavor=flv,
                df=df,
                bbox=bbox,
                area_ratio=ar,
                density=_content_density(df),
                rows=df.shape[0],
                cols=df.shape[1]
            ))
    return candidates

def _is_credible_table(c: Dict[str, Any],
                       min_cols: int = 5,
                       min_rows: int = 6,
                       min_area_ratio: float = 0.10,
                       max_area_ratio: float = 0.90,
                       max_density: float = 0.95) -> bool:
    return (
        c["cols"] >= min_cols and
        c["rows"] >= min_rows and
        min_area_ratio <= c["area_ratio"] <= max_area_ratio and
        c["density"] <= max_density
    )

# -----------------------------
# RAG (with retry) + graceful fallbacks
# -----------------------------
def extract_pages_text(pdf_path: str) -> List[str]:
    texts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            txt = page.get_text("text") or ""
            texts.append(txt)
    return texts

def _to_numpy_embedding(resp) -> np.ndarray:
    emb = resp.get("embedding")
    values = emb["values"] if isinstance(emb, dict) and "values" in emb else emb
    return np.array(values, dtype="float32")

def embed_text_safe(text: str, task_type: str, max_retries: int = 3, backoff: float = 1.5):
    last_err = None
    for i in range(max_retries):
        try:
            resp = genai.embed_content(model=EMBED_MODEL, content=text, task_type=task_type)
            return _to_numpy_embedding(resp)
        except Exception as e:
            last_err = e
            time.sleep(backoff ** i)
    raise last_err

def find_anchor_page_rag(pdf_path: str, query: str) -> Optional[int]:
    if not API_KEY:
        return None
    pages = extract_pages_text(pdf_path)
    if not pages:
        return None
    try:
        embs = np.vstack([embed_text_safe(p or "empty page", "retrieval_document") for p in pages]).astype("float32")
        index = faiss.IndexFlatL2(embs.shape[1])
        index.add(embs)
        q = embed_text_safe(query, "retrieval_query").reshape(1, -1).astype("float32")
        D, I = index.search(q, k=1)
        return int(I[0][0]) + 1
    except Exception:
        return None  # graceful fallback on any API/server error

def find_anchor_page_keywords(pdf_path: str, keywords: List[str]) -> Optional[int]:
    pages = extract_pages_text(pdf_path)
    for i, txt in enumerate(pages, start=1):
        t = txt.lower()
        if any(kw.lower() in t for kw in keywords):
            return i
    return None

def find_anchor_page_camelot(pdf_path: str, prefer_stream: bool = True) -> Optional[int]:
    # Scan document for first credible multi-column table
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
    for p in range(1, total_pages + 1):
        cands = _collect_candidates(pdf_path, p, prefer_stream=prefer_stream)
        cred = [c for c in cands if _is_credible_table(c)]
        if cred:
            return p
    return None

def find_anchor_page(pdf_path: str, query: str, keywords: Optional[List[str]] = None,
                     prefer_stream: bool = True) -> Optional[int]:
    # Try RAG, then keywords, then Camelot-only
    anchor = find_anchor_page_rag(pdf_path, query)
    if anchor:
        return anchor
    if keywords:
        anchor = find_anchor_page_keywords(pdf_path, keywords)
        if anchor:
            return anchor
    return find_anchor_page_camelot(pdf_path, prefer_stream=prefer_stream)

# -----------------------------
# Pages-only finder (RAG for pages, Camelot for structure)
# -----------------------------
def get_table_pages_list_rag(
    pdf_path: str,
    query: str,
    keywords: Optional[List[str]] = None,
    max_lookahead: int = 8,
    prefer_stream: bool = True,
    area_pad: float = 18.0,
    density_threshold: float = 0.10,
    schema_tolerance: int = 1,
    debug: bool = False,
) -> List[int]:
    anchor_page = find_anchor_page(pdf_path, query, keywords=keywords, prefer_stream=prefer_stream)
    if anchor_page is None:
        return []

    # Re-anchor within a small forward window to avoid headings
    window_pages = list(range(anchor_page, anchor_page + 3))
    start_page = None
    anchor_cols = None
    anchor_bbox = None

    for p in window_pages:
        cands = _collect_candidates(pdf_path, p, prefer_stream=prefer_stream)
        cred = [c for c in cands if _is_credible_table(c)]
        if debug:
            print(f"[debug] Page {p} candidates={len(cands)} credible={len(cred)}")
        if cred:
            best = max(cred, key=lambda c: (c["density"], c["rows"]))
            start_page = p
            anchor_cols = best["cols"]
            anchor_bbox = best["bbox"]
            break

    if start_page is None:
        return [anchor_page]

    area_bl = None
    area_tl = None
    if anchor_bbox:
        page_w, page_h = _page_size(pdf_path, start_page)
        area_bl, area_tl = _bbox_to_areas(anchor_bbox, page_h)
        area_bl = _expand_area(area_bl, area_pad, page_w, page_h)
        area_tl = _expand_area(area_tl, area_pad, page_w, page_h)

    pages = [start_page]

    # Scan forward for continuations using structure only
    for page in range(start_page + 1, start_page + 1 + max_lookahead):
        found = False
        for flavor in (["stream", "lattice"] if prefer_stream else ["lattice", "stream"]):
            candidates: List[pd.DataFrame] = []

            # Locked regions first
            for area in [area_bl, area_tl]:
                if not area:
                    continue
                t = _read_tables(pdf_path, page, flavor=flavor, table_areas=[area])
                if t and t.n > 0:
                    dfs = []
                    for tbl in t:
                        df = _clean_df(tbl.df)
                        if df is not None and not df.empty and abs(df.shape[1] - anchor_cols) <= schema_tolerance:
                            dfs.append(df)
                    if dfs:
                        candidates.append(max(dfs, key=_content_density))

            # Whole page fallback
            if not candidates:
                t = _read_tables(pdf_path, page, flavor=flavor)
                if t and t.n > 0:
                    dfs = []
                    for tbl in t:
                        df = _clean_df(tbl.df)
                        if df is not None and not df.empty and abs(df.shape[1] - anchor_cols) <= schema_tolerance:
                            dfs.append(df)
                    if dfs:
                        candidates.append(max(dfs, key=_content_density))

            for df_cand in candidates:
                dens = _content_density(df_cand)
                if debug:
                    print(f"[debug] Page {page} {flavor} cols={df_cand.shape[1]} dens={dens:.3f}")
                if dens >= density_threshold:
                    pages.append(page)
                    found = True
                    break

            if found:
                break

        if not found:
            break

    return pages

# -----------------------------
# Example (pages only)
# -----------------------------
if __name__ == "__main__":
    pdf_path = "Prot_000.pdf"
    # Optional keywords to assist fallback if RAG fails
    keywords = ["Time and Events Schedule", "Schedule of Activities", "Time and Events"]

    pages = get_table_pages_list_rag(
        pdf_path=pdf_path,
        query="Time and Events Schedule",
        keywords=keywords,
        max_lookahead=8,
        prefer_stream=True,
        area_pad=18.0,
        density_threshold=0.10,
        schema_tolerance=1,
        debug=True
    )
    print(f"Detected table pages: {pages}")
