#!/usr/bin/env python3
"""
Hard-coded: PDF (with OCR fallback) → Gemini embeddings → FAISS vector DB → Visual HTML report
- Extract tables (patients/trials) from PDF (Camelot/Tabula/pdfplumber)
- Fallback to OCR for scanned PDFs (pdf2image + Tesseract)
- Clean & unify tables
- Embed rows with Gemini
- Store vectors + metadata in FAISS
- Visualize 2D scatter + interactive table
- Generate narrative analysis with Gemini
- Export self-contained HTML report

Note: This report is informational and not medical advice.
"""

import os, json, base64, re
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
import plotly.graph_objects as go
from jinja2 import Template
import faiss
import google.generativeai as genai
from dotenv import load_dotenv

# -----------------------
# HARD-CODED SETTINGS (edit these)
# -----------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
PDF_PATH   = r"C:\Users\2000171723\Documents\Clinical_trial\Prot_0.pdf"  # <-- set your PDF path
OUTPUT_DIR = r"C:\Users\2000171723\Documents\Clinical_trial\output"         # <-- set your output folder
COLOR_BY   = "Outcome"  # optional: column for coloring scatter plot (e.g., "Outcome", "Arm", "Site")
SUMMARY_MODEL = "gemini-1.5-flash"
EMBED_MODEL   = "text-embedding-004"

# OCR fallback dependencies
HAS_PDF2IMAGE = False
HAS_TESSERACT = False
try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except Exception:
    pass
try:
    import pytesseract
    HAS_TESSERACT = True
except Exception:
    pass

# PDF extraction backends
HAS_CAM = HAS_TAB = HAS_PLUM = False
try:
    import camelot  # requires Ghostscript
    HAS_CAM = True
except Exception:
    pass
try:
    import tabula  # requires Java
    HAS_TAB = True
except Exception:
    pass
try:
    import pdfplumber
    HAS_PLUM = True
except Exception:
    pass


# -----------------------
# PDF table extraction (non-OCR)
# -----------------------
def extract_tables_pdf(pdf_path: str) -> List[pd.DataFrame]:
    tables: List[pd.DataFrame] = []

    # Camelot: try lattice then stream
    if HAS_CAM:
        try:
            for flavor in ["lattice", "stream"]:
                cam_tabs = camelot.read_pdf(pdf_path, flavor=flavor, pages="all")
                for t in cam_tabs:
                    df = t.df
                    if df is not None and df.shape[0] > 0 and df.shape[1] > 0:
                        tables.append(df)
                if tables:
                    return tables
        except Exception:
            pass

    # Tabula
    if HAS_TAB:
        try:
            dfs = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)
            for df in dfs:
                if df is not None and df.shape[0] > 0 and df.shape[1] > 0:
                    tables.append(df)
            if tables:
                return tables
        except Exception:
            pass

    # pdfplumber
    if HAS_PLUM:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    for tbl in page.extract_tables():
                        df = pd.DataFrame(tbl)
                        if df.shape[0] > 0 and df.shape[1] > 0:
                            tables.append(df)
            if tables:
                return tables
        except Exception:
            pass

    return tables


# -----------------------
# OCR fallback: extract tables from scanned PDFs
# Approach:
# - Render pages to images via pdf2image
# - Use Tesseract to get words per line
# - Split lines into columns by multiple spaces (heuristic)
# -----------------------
def ocr_pages_to_lines(pdf_path: str, dpi: int = 300) -> List[List[str]]:
    if not (HAS_PDF2IMAGE and HAS_TESSERACT):
        raise RuntimeError("OCR fallback unavailable: install pdf2image (Poppler) and Tesseract.")

    images = convert_from_path(pdf_path, dpi=dpi)
    all_lines: List[List[str]] = []

    for img in images:
        # Get per-line text via Tesseract (PSM 6 = assume a single uniform block of text)
        text = pytesseract.image_to_string(img, config="--psm 6")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            all_lines.append(lines)

    return all_lines


def parse_lines_to_table(lines_per_page: List[List[str]]) -> List[pd.DataFrame]:
    # Heuristic: split by runs of 2+ spaces to form columns; infer max columns across lines
    dfs: List[pd.DataFrame] = []
    for lines in lines_per_page:
        split_lines = [re.split(r"\s{2,}", ln.strip()) for ln in lines]
        max_cols = max((len(s) for s in split_lines if s), default=0)
        if max_cols < 2:
            # Not tabular enough; skip
            continue
        # Normalize row lengths
        normalized = [s + [""] * (max_cols - len(s)) for s in split_lines]
        df = pd.DataFrame(normalized)
        # Try to treat first row as header if strings unique
        if df.shape[0] > 1:
            first = df.iloc[0].astype(str).tolist()
            if len(set(first)) == len(first):
                df.columns = [str(x).strip() or f"col_{i}" for i, x in enumerate(first)]
                df = df.iloc[1:].reset_index(drop=True)
            else:
                df.columns = [f"col_{i}" for i in range(df.shape[1])]
        else:
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
        dfs.append(df)

    return dfs


# -----------------------
# Cleaning & unification
# -----------------------
def clean_and_unify_tables(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        raise RuntimeError("No tables found. If this PDF is scanned, ensure OCR fallback is enabled.")

    cleaned: List[pd.DataFrame] = []
    for df in dfs:
        df = df.copy()
        # Promote first row to header if unique
        try:
            first_row = df.iloc[0].astype(str).tolist()
            header_like = len(set(first_row)) == len(first_row)
        except Exception:
            header_like = False

        if header_like:
            df.columns = [str(c).strip() for c in first_row]
            df = df.iloc[1:].reset_index(drop=True)
        else:
            df.columns = [f"col_{i}" for i in range(df.shape[1])]

        # Strip whitespace
        df = df.applymap(lambda x: str(x).strip() if pd.notna(x) else x)
        # Drop rows with all empty
        df = df.replace({"": np.nan, "None": np.nan, "nan": np.nan}).dropna(how="all")
        # Keep non-empty columns
        df = df[[c for c in df.columns if not all(df[c].astype(str).str.strip().isin(["", "None", "nan"]))]]
        if df.shape[0] > 0 and df.shape[1] > 0:
            cleaned.append(df)

    if not cleaned:
        raise RuntimeError("Tables were extracted but none were usable after cleaning.")

    # Align columns across tables via union
    base_cols = cleaned[0].columns.tolist()
    for i in range(1, len(cleaned)):
        if cleaned[i].columns.tolist() != base_cols:
            base_cols = list(dict.fromkeys(base_cols + cleaned[i].columns.tolist()))

    aligned = []
    for df in cleaned:
        for col in base_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[base_cols]
        aligned.append(df)

    unified = pd.concat(aligned, axis=0, ignore_index=True)
    unified = unified.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    unified = unified.dropna(how="all").reset_index(drop=True)
    return unified


# -----------------------
# Column typing
# -----------------------
def detect_column_types(df: pd.DataFrame):
    numeric_cols, categorical_cols, text_cols = [], [], []
    for col in df.columns:
        s = df[col]
        try:
            pd.to_numeric(s, errors="raise")
            numeric_cols.append(col)
            continue
        except Exception:
            pass
        uniques = s.dropna().unique()
        if len(uniques) <= max(10, int(0.05 * len(s))):
            categorical_cols.append(col)
        else:
            text_cols.append(col)
    return numeric_cols, categorical_cols, text_cols


# -----------------------
# Row → text and embeddings (Gemini)
# -----------------------
def row_to_text(row: pd.Series) -> str:
    parts = []
    for col, val in row.items():
        if pd.isna(val):
            continue
        s = str(val).strip()
        if s:
            parts.append(f"{col}: {s}")
    return "; ".join(parts)


def setup_gemini():
    if not API_KEY:
        raise RuntimeError("Gemini API key missing. Set API_KEY at top of script.")
    genai.configure(api_key=API_KEY)


def gemini_embed_texts(texts, embed_model=EMBED_MODEL):
    setup_gemini()
    embeddings = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = genai.embed_content(model=embed_model, content=batch)

        # New client returns resp.embeddings (list of lists of floats) for batches
        if hasattr(resp, "embeddings") and resp.embeddings:
            vecs = resp.embeddings
        elif hasattr(resp, "embedding") and resp.embedding:
            vecs = [resp.embedding]
        else:
            raise RuntimeError(f"Unexpected embedding response: {resp}")

        embeddings.extend([np.array(v, dtype=np.float32) for v in vecs])

    arr = np.vstack(embeddings)
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def gemini_generate_summary(df: pd.DataFrame,
                            numeric_cols: List[str],
                            categorical_cols: List[str],
                            text_cols: List[str],
                            model_name: str = SUMMARY_MODEL) -> str:
    setup_gemini()
    model = genai.GenerativeModel(model_name)
    sample_rows = df.head(100).to_dict(orient="records")
    prompt = (
        "You are analyzing a table of patient/trial records extracted from a PDF. "
        "Provide a concise, clear report with:\n"
        "- What the table seems to represent and key columns\n"
        "- Notable distributions, ranges, and categorical breakdowns\n"
        "- Any patterns, clusters, or outliers (based on column types)\n"
        "- Caveats (missing data, mixed datatypes, possible OCR errors)\n"
        "Keep it professional and avoid medical advice.\n\n"
        f"Columns: {list(df.columns)}\n"
        f"Numeric columns: {numeric_cols}\n"
        f"Categorical columns: {categorical_cols}\n"
        f"Text columns: {text_cols}\n"
        f"Sample (up to 100 rows): {json.dumps(sample_rows, ensure_ascii=False)[:12000]}"
    )
    resp = model.generate_content(prompt)
    return resp.text or "Summary unavailable."


# -----------------------
# FAISS index
# -----------------------
def build_faiss_index(embeddings: np.ndarray):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype(np.float32))
    return index


# -----------------------
# Dimensionality reduction & viz
# -----------------------
def reduce_dimensions(embeddings: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, str]:
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=random_state)
        coords = reducer.fit_transform(embeddings)
        return coords, "UMAP"
    except Exception:
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(embeddings)
        return coords, "PCA"


def fig_to_base64_png(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_scatter_plot(df: pd.DataFrame, coords: np.ndarray, color_col: Optional[str]) -> str:
    plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    fig, ax = plt.subplots(figsize=(8, 6))
    if color_col and color_col in df.columns:
        plot_df[color_col] = df[color_col].astype(str)
        sns.scatterplot(data=plot_df, x="x", y="y", hue=color_col, s=35, ax=ax, palette="tab10")
        ax.legend(loc="best", title=color_col, fontsize=8)
    else:
        sns.scatterplot(data=plot_df, x="x", y="y", s=35, ax=ax, color="#1f77b4")
    ax.set_title("Embedding scatter (2D)")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    b64 = fig_to_base64_png(fig)
    plt.close(fig)
    return b64


def build_table_plotly_html(df: pd.DataFrame, max_rows: int = 1500) -> str:
    display_df = df.copy()
    if display_df.shape[0] > max_rows:
        display_df = display_df.head(max_rows)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(display_df.columns), fill_color='lightgrey', align='left'),
        cells=dict(values=[display_df[col].astype(str).tolist() for col in display_df.columns],
                   fill_color='white', align='left')
    )])
    fig.update_layout(width=1100, height=min(900, 260 + 20 * display_df.shape[0]))
    return fig.to_html(include_plotlyjs='cdn', full_html=False)


# -----------------------
# Report assembly
# -----------------------
def build_report_html(
    df: pd.DataFrame,
    stats: dict,
    color_col: Optional[str],
    method_label: str,
    scatter_png_b64: str,
    table_html_fragment: str,
    narrative: str
) -> str:
    template = Template("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Patient/Trial Table Report</title>
<style>
body { font-family: Arial, sans-serif; margin: 24px; color: #222; }
h1, h2, h3 { margin-top: 0.6em; }
.section { margin-bottom: 24px; }
.kv { margin: 4px 0; }
.kv b { display: inline-block; width: 180px; }
img { max-width: 100%; height: auto; border: 1px solid #ddd; }
.code { background: #f7f7f9; padding: 10px; border: 1px solid #eee; white-space: pre-wrap; }
.footer { color: #777; font-size: 12px; margin-top: 30px; }
.note { color: #555; font-size: 13px; }
</style>
</head>
<body>
<h1>Patient/Trial Table Report</h1>

<div class="section">
  <h2>Summary</h2>
  <div class="kv"><b>Rows:</b> {{ stats.rows }}</div>
  <div class="kv"><b>Columns:</b> {{ stats.cols }}</div>
  <div class="kv"><b>Color by:</b> {{ color_col if color_col else "N/A" }}</div>
  <div class="kv"><b>Dimensionality reduction:</b> {{ method_label }}</div>
  <p class="note">This report summarizes patterns in the table and shows a 2D projection of semantic similarity between records. It is informational and not medical advice.</p>
</div>

<div class="section">
  <h2>Embedding scatter (2D)</h2>
  <p>Nearby points indicate records with similar textual descriptions or attributes after embedding.</p>
  <img src="data:image/png;base64,{{ scatter_png_b64 }}" alt="Embedding scatter">
</div>

<div class="section">
  <h2>Interactive table preview</h2>
  {{ table_html_fragment | safe }}
</div>

<div class="section">
  <h2>Narrative analysis</h2>
  <div class="code">{{ narrative }}</div>
</div>

<div class="section">
  <h2>Column dtypes</h2>
  <div class="code">{{ col_info }}</div>
</div>

<div class="footer">
  Generated automatically. For large tables, the preview may be truncated.
</div>
</body>
</html>
    """)
    col_info = json.dumps({c: str(df[c].dtype) for c in df.columns}, indent=2)
    html = template.render(
        stats=stats,
        color_col=color_col,
        method_label=method_label,
        scatter_png_b64=scatter_png_b64,
        table_html_fragment=table_html_fragment,
        narrative=narrative,
        col_info=col_info
    )
    return html


# -----------------------
# Utility
# -----------------------
def choose_color_column(df: pd.DataFrame, categorical_cols: List[str], user_choice: Optional[str]) -> Optional[str]:
    if user_choice and user_choice in df.columns:
        return user_choice
    if categorical_cols:
        return categorical_cols[0]
    # fallback: any column with < 30 unique values
    candidates = []
    for col in df.columns:
        uniques = df[col].dropna().unique()
        if len(uniques) < 30:
            candidates.append(col)
    return candidates[0] if candidates else None


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Try direct PDF table extraction
    dfs = extract_tables_pdf(PDF_PATH)

    # 2) If none, OCR fallback
    if not dfs:
        print("No machine-readable tables found. Trying OCR fallback...")
        lines_pages = ocr_pages_to_lines(PDF_PATH)
        dfs_ocr = parse_lines_to_table(lines_pages)
        if not dfs_ocr:
            raise RuntimeError("OCR fallback could not detect tabular content. "
                               "Consider improving scan quality or providing a digital PDF.")
        dfs = dfs_ocr

    # 3) Clean and unify tables
    df = clean_and_unify_tables(dfs)
    df = df.dropna(how="all").reset_index(drop=True)

    # 4) Detect column types
    numeric_cols, categorical_cols, text_cols = detect_column_types(df)

    # 5) Row texts and embeddings
    texts = [row_to_text(df.iloc[i]) for i in range(df.shape[0])]
    embeddings = gemini_embed_texts(texts)

    # 6) Vector DB (FAISS) + metadata
    index = build_faiss_index(embeddings)
    faiss.write_index(index, os.path.join(OUTPUT_DIR, "faiss.index"))
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_table.csv"), index=False)

    # 7) Dimensionality reduction + viz
    coords, method_label = reduce_dimensions(embeddings)
    chosen_color = choose_color_column(df, categorical_cols, COLOR_BY)
    scatter_png_b64 = build_scatter_plot(df, coords, chosen_color)
    table_html_fragment = build_table_plotly_html(df)

    # 8) Narrative analysis
    narrative = gemini_generate_summary(df, numeric_cols, categorical_cols, text_cols)

    # 9) Report assembly
    stats = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    report_html = build_report_html(
        df=df,
        stats=stats,
        color_col=chosen_color,
        method_label=method_label,
        scatter_png_b64=scatter_png_b64,
        table_html_fragment=table_html_fragment,
        narrative=narrative
    )
    report_path = os.path.join(OUTPUT_DIR, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)

    print("Done.")
    print(f"Saved:\n- {report_path}\n- {os.path.join(OUTPUT_DIR, 'faiss.index')}\n- {os.path.join(OUTPUT_DIR, 'metadata.json')}\n- {os.path.join(OUTPUT_DIR, 'cleaned_table.csv')}")
