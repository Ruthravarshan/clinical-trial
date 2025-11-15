# language: python
import os
import re
import fitz
import faiss
import numpy as np
import camelot
import pandas as pd
import google.generativeai as genai
from typing import List, Tuple, Optional
from dotenv import load_dotenv

# ==============================
# Configuration
# ==============================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

EMBED_MODEL = "text-embedding-004"

# Camelot tuning defaults (you can tweak if needed)
CAMELOT_KW_DEFAULTS = dict(
    strip_text=" \n",
    # Try stronger line detection; adjust if needed
    line_scale=40,
)

# ==============================
# Embedding helpers (for locating the start page)
# ==============================
def extract_pages(pdf_path: str) -> List[str]:
    texts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            txt = page.get_text("text") or ""
            if not txt.strip():
                txt = "empty page"
            texts.append(txt)
    return texts

def _to_numpy_embedding(resp) -> np.ndarray:
    emb = resp.get("embedding")
    values = emb["values"] if isinstance(emb, dict) and "values" in emb else emb
    return np.array(values, dtype="float32")

def embed_text(text: str, task_type: str = "retrieval_document") -> np.ndarray:
    resp = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type=task_type
    )
    return _to_numpy_embedding(resp)

def embed_pages(pages: List[str]) -> np.ndarray:
    vectors = []
    for page_text in pages:
        vec = embed_text(page_text, task_type="retrieval_document")
        vectors.append(vec)
    return np.vstack(vectors).astype("float32")

def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def find_schedule_page(index: faiss.IndexFlatL2, query: str, k: int = 1) -> int:
    query_vec = embed_text(query, task_type="retrieval_query").reshape(1, -1).astype("float32")
    D, I = index.search(query_vec, k=k)
    return int(I[0][0]) + 1  # return 1-indexed page number

def get_schedule_page(pdf_path: str, query: str = "Time and Events Schedule") -> int:
    pages = extract_pages(pdf_path)
    if len(pages) == 0:
        raise ValueError("No pages extracted from the PDF.")
    embeddings = embed_pages(pages)
    index = build_index(embeddings)
    page_number = find_schedule_page(index, query=query, k=1)
    return page_number +1 # already 1-indexed

# ==============================
# Camelot helpers
# ==============================
def read_camelot_table(
    pdf_path: str,
    page_number: int,
    table_areas: Optional[List[str]] = None,
    prefer_flavor: str = "lattice",
) -> Optional[camelot.core.Table]:
    """
    Try to read a single table from a given page, first with `prefer_flavor`,
    then fallback to the other flavor if nothing found. Returns the first table or None.
    """
    flavors = [prefer_flavor, "stream" if prefer_flavor == "lattice" else "lattice"]
    for fl in flavors:
        kw = dict(
            pages=str(page_number),
            flavor=fl,
        )
        kw.update(CAMELOT_KW_DEFAULTS)
        if table_areas:
            kw["table_areas"] = table_areas
        try:
            tables = camelot.read_pdf(pdf_path, **kw)
        except Exception:
            tables = None
        if tables and tables.n > 0:
            return tables[0]
    return None

def clean_cell_text(txt: str) -> str:
    if txt is None:
        return ""
    # Replace newlines with space, collapse multiple spaces
    txt = txt.replace("\n", " ").strip()
    txt = re.sub(r"\s+", " ", txt)
    return txt

def table_cells_text_row(table: camelot.core.Table, row_idx: int) -> List[str]:
    row = table.cells[row_idx]
    return [clean_cell_text(c.text) for c in row]

def compute_table_bbox(table: camelot.core.Table, pad: float = 3.0) -> Tuple[float, float, float, float]:
    xs = []
    ys = []
    for row in table.cells:
        for c in row:
            xs.extend([c.x1, c.x2])
            ys.extend([c.y1, c.y2])
    if not xs or not ys:
        raise ValueError("Could not compute table bbox; empty cell coordinates.")
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    # Expand slightly
    return (max(min_x - pad, 0), max(min_y - pad, 0), max_x + pad, max_y + pad)

def bbox_to_table_areas(bbox: Tuple[float, float, float, float]) -> List[str]:
    x1, y1, x2, y2 = bbox
    return [f"{x1},{y1},{x2},{y2}"]

def find_subheader_row_index(
    table: camelot.core.Table,
    ncols: int,
    max_scan_rows: int = 6,
    min_fill_ratio: float = 0.6
) -> Optional[int]:
    """
    Heuristic: the subheader row is the first row near the top where a majority of columns are filled.
    """
    limit = min(max_scan_rows, len(table.cells))
    for r in range(limit):
        texts = table_cells_text_row(table, r)
        filled = sum(1 for t in texts if t.strip() != "")
        if filled >= int(ncols * min_fill_ratio):
            return r
    return None

def find_top_header_row_index(
    table: camelot.core.Table,
    subheader_row_idx: int
) -> Optional[int]:
    """
    Heuristic: top header is the nearest row above subheader that has any non-empty text.
    """
    for r in range(subheader_row_idx - 1, -1, -1):
        texts = table_cells_text_row(table, r)
        if any(t.strip() != "" for t in texts):
            return r
    return None

def get_col_spans_from_row(table: camelot.core.Table, row_idx: int) -> List[Tuple[float, float]]:
    row = table.cells[row_idx]
    spans = [(c.x1, c.x2) for c in row]
    # Ensure left-to-right sorted by center x; Camelot generally keeps order, but be safe
    spans = sorted(spans, key=lambda ab: (ab[0] + ab[1]) / 2.0)
    return spans

def forward_fill_headers(header_cells: List[str]) -> List[str]:
    """
    Forward-fill empty header cells with the last non-empty header text.
    """
    out = []
    last = ""
    for t in header_cells:
        if t.strip():
            last = t.strip()
            out.append(last)
        else:
            out.append(last)
    return out

def build_final_headers(top_headers: List[str], sub_headers: List[str]) -> List[str]:
    final = []
    for top, sub in zip(top_headers, sub_headers):
        top = top.strip()
        sub = sub.strip()
        if top and sub:
            final.append(f"{top} | {sub}")
        elif sub:
            final.append(sub)
        elif top:
            final.append(top)
        else:
            final.append("")
    return final

def spans_center(sp: Tuple[float, float]) -> float:
    return (sp[0] + sp[1]) / 2.0

def map_local_to_anchor_cols(
    local_spans: List[Tuple[float, float]],
    anchor_spans: List[Tuple[float, float]]
) -> List[int]:
    """
    For each local column span, find the anchor column index by center containment;
    if not contained, pick nearest anchor center.
    Returns mapping list m where m[i_local] = j_anchor.
    """
    anchor_centers = [spans_center(s) for s in anchor_spans]
    mapping = []
    for (lx1, lx2) in local_spans:
        cx = (lx1 + lx2) / 2.0
        # Try containment first
        candidates = [j for j, (ax1, ax2) in enumerate(anchor_spans) if ax1 <= cx <= ax2]
        if candidates:
            mapping.append(candidates[0])
        else:
            # Nearest by center distance
            dists = [abs(cx - ac) for ac in anchor_centers]
            jmin = int(np.argmin(dists))
            mapping.append(jmin)
    return mapping

def reorder_columns_by_mapping(df: pd.DataFrame, mapping_local_to_anchor: List[int]) -> pd.DataFrame:
    """
    Reorder df columns so that new order matches anchor column order.
    mapping[i_local] = j_anchor. We build inverse mapping: for each anchor j, find i_local with that j.
    """
    ncols = len(mapping_local_to_anchor)
    inv = [None] * ncols
    for i_local, j_anchor in enumerate(mapping_local_to_anchor):
        # If duplicates happen, keep the first occurrence
        if 0 <= j_anchor < ncols and inv[j_anchor] is None:
            inv[j_anchor] = i_local
    # Fill any missing by stable order fallback
    remaining = [i for i in range(ncols) if i not in inv]
    for j in range(ncols):
        if inv[j] is None:
            inv[j] = remaining.pop(0)
    return df.iloc[:, inv]

# ==============================
# Multi-page table extraction
# ==============================
def extract_anchor_table_info(table: camelot.core.Table) -> dict:
    """
    Extract anchor info: ncols, bbox, anchor spans, header rows and final headers, and data frame for anchor page.
    """
    df = table.df.copy()
    ncols = df.shape[1]

    # Identify subheader row (per-column header), then top header row (merged header) if present
    sub_idx = find_subheader_row_index(table, ncols=ncols, max_scan_rows=6)
    if sub_idx is None:
        # Fallback: assume first row is subheader
        sub_idx = 0

    top_idx = find_top_header_row_index(table, subheader_row_idx=sub_idx)

    sub_headers = table_cells_text_row(table, sub_idx)
    if top_idx is not None:
        raw_top_headers = table_cells_text_row(table, top_idx)
        top_headers = forward_fill_headers(raw_top_headers)
    else:
        top_headers = [""] * ncols

    final_headers = build_final_headers(top_headers, sub_headers)

    # Column spans from the subheader row
    col_spans = get_col_spans_from_row(table, sub_idx)

    # Compute bounding box
    bbox = compute_table_bbox(table, pad=4.0)  # slightly larger pad

    # Anchor data rows start after subheader
    data_start = sub_idx + 1
    df_data = df.iloc[data_start:].reset_index(drop=True)
    df_data.columns = final_headers

    return dict(
        ncols=ncols,
        bbox=bbox,
        anchor_spans=col_spans,
        top_headers=top_headers,
        sub_headers=sub_headers,
        final_headers=final_headers,
        data=df_data,
        sub_idx=sub_idx,
        top_idx=top_idx,
    )

def extract_continuation_page(
    pdf_path: str,
    page_number: int,
    anchor_info: dict,
) -> Optional[pd.DataFrame]:
    """
    Extract table data from a continuation page using anchor info.
    Returns a DataFrame or None if nothing valid found.
    """
    area = bbox_to_table_areas(anchor_info["bbox"])
    table = read_camelot_table(pdf_path, page_number, table_areas=area, prefer_flavor="lattice")
    if table is None:
        return None

    df = table.df.copy()
    if df.shape[1] != anchor_info["ncols"]:
        # Attempt flavor fallback already done; reject if column count differs too much
        return None

    # Identify subheader row on continuation page
    ncols = anchor_info["ncols"]
    sub_idx = find_subheader_row_index(table, ncols=ncols, max_scan_rows=6)
    if sub_idx is None:
        # If we can't find a header row, abort this page
        return None

    sub_headers = table_cells_text_row(table, sub_idx)
    local_spans = get_col_spans_from_row(table, sub_idx)
    mapping = map_local_to_anchor_cols(local_spans, anchor_info["anchor_spans"])

    # Reorder columns to match anchor order
    df_data = df.iloc[sub_idx + 1:].reset_index(drop=True)
    df_data = reorder_columns_by_mapping(df_data, mapping)

    # Build final headers for this page using anchor top headers + this page's subheaders reordered
    # First, reorder sub_headers according to anchor mapping
    # mapping[i_local] = j_anchor => we want sub_headers ordered by j_anchor ascending
    sub_by_anchor = [None] * ncols
    for i_local, j_anchor in enumerate(mapping):
        if 0 <= j_anchor < ncols and sub_by_anchor[j_anchor] is None:
            sub_by_anchor[j_anchor] = sub_headers[i_local]
    # Fill any missing with blanks
    for j in range(ncols):
        if sub_by_anchor[j] is None:
            sub_by_anchor[j] = ""

    page_headers = build_final_headers(anchor_info["top_headers"], sub_by_anchor)
    df_data.columns = page_headers
    return df_data

def extract_multipage_table_from_start(
    pdf_path: str,
    start_page: int,
    max_lookahead: int = 10
) -> Optional[pd.DataFrame]:
    """
    Extracts a multi-page table starting at start_page and continuing forward until no table found.
    Returns a concatenated DataFrame or None if nothing extracted.
    """
    # Anchor page
    anchor_table = read_camelot_table(pdf_path, start_page, prefer_flavor="lattice")
    if anchor_table is None:
        # Fallback to stream already handled; give up if still None
        print(f"[WARN] No table found on anchor page {start_page}.")
        return None

    anchor_info = extract_anchor_table_info(anchor_table)
    all_chunks = [anchor_info["data"]]

    # Iterate subsequent pages
    for p in range(start_page + 1, start_page + 1 + max_lookahead):
        chunk = extract_continuation_page(pdf_path, p, anchor_info)
        if chunk is None or chunk.empty:
            print(f"[INFO] Stopping at page {p}: no valid continuation table found.")
            break
        print(f"[INFO] Appending continuation from page {p} with shape {chunk.shape}.")
        all_chunks.append(chunk)

    if not all_chunks:
        return None
    df_all = pd.concat(all_chunks, ignore_index=True)
    return df_all

# ==============================
# End-to-end runner
# ==============================
def extract_schedule_table(
    pdf_path: str,
    query: str = "Time and Events Schedule",
    start_page: Optional[int] = None,
    max_lookahead: int = 10
) -> Optional[pd.DataFrame]:
    """
    End-to-end:
    - If start_page not provided, uses embeddings to find the best matching start page.
    - Extracts a multi-page table starting at that page.
    """
    if start_page is None:
        if not API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set; either set it or pass start_page explicitly.")
        start_page = get_schedule_page(pdf_path, query=query)
        print(f"[INFO] Detected anchor start page: {start_page}")
    else:
        print(f"[INFO] Using provided start page: {start_page}")

    # Important: start_page is already 1-indexed; do NOT add +1 here
    df_all = extract_multipage_table_from_start(pdf_path, start_page=start_page, max_lookahead=max_lookahead)
    return df_all

# ==============================
# Example main
# ==============================
if __name__ == "__main__":
    pdf_path = "Prot_000.pdf"
    # Option A: Let the model find the page (requires GEMINI_API_KEY)
    try:
        df = extract_schedule_table(pdf_path, query="Time and Events Schedule", start_page=None, max_lookahead=10)
    except Exception as e:
        print(f"[WARN] Embedding-based page detection failed: {e}")
        print("[INFO] Falling back to manual start_page=1 (change as needed).")
        df = extract_schedule_table(pdf_path, start_page=1, max_lookahead=10)

    if df is not None and not df.empty:
        print("[OK] Multi-page table extracted successfully.")
        print(df.head(10))
        # Save for further processing
        df.to_pickle("my_dataframe.pkl")
        df.to_csv("my_dataframe.csv", index=False)
    else:
        print("[INFO] No multi-page table extracted.")
