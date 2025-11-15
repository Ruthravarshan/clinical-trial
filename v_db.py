#!/usr/bin/env python3
"""
PDF → Gemini embeddings → FAISS vector DB → Visual HTML report
- Extract tables (patients/trials) from PDF
- Embed rows with Gemini
- Store vectors + metadata
- Visualize embeddings + interactive table
- Generate narrative analysis using Gemini
- Export self-contained HTML report

Usage:
  python pdf_gemini_vector_report.py \
    --pdf_path path/to/file.pdf \
    --output_dir ./output \
    --color_by ColumnName \
    [--api_key YOUR_KEY] \
    [--summary_model gemini-1.5-flash] \
    [--embed_model text-embedding-004]
"""

import os
import sys
import json
import base64
import argparse
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# -----------------------
# HARD-CODED SETTINGS
# -----------------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
PDF_PATH  = r"C:\Users\2000171723\Documents\Clinical_trial\Prot_0.pdf"  # <--- your PDF path
OUTPUT_DIR = r"C:\Users\2000171723\Documents\Clinical_trial\output"
COLOR_BY  = "Outcome"   # optional column name to color scatter plot
SUMMARY_MODEL = "gemini-1.5-flash"
EMBED_MODEL   = "text-embedding-004"

# PDF extraction backends
EXTRACT_BACKENDS = []
try:
    import camelot  # requires ghostscript
    EXTRACT_BACKENDS.append("camelot")
except Exception:
    pass

try:
    import tabula  # requires Java
    EXTRACT_BACKENDS.append("tabula")
except Exception:
    pass

try:
    import pdfplumber
    EXTRACT_BACKENDS.append("pdfplumber")
except Exception:
    pass

# Gemini (Google Generative AI)
import google.generativeai as genai

# Vector DB (local)
import faiss

# Visualization
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# Plotly for interactive table
import plotly.graph_objects as go

# Templating for report
from jinja2 import Template


# -----------------------
# PDF Extraction
# -----------------------
def extract_tables(pdf_path: str, max_pages: Optional[int] = None) -> List[pd.DataFrame]:
    """
    Extract tables from a PDF using available backends.
    Returns list of DataFrames.
    """
    tables = []

    if "camelot" in EXTRACT_BACKENDS:
        try:
            for flavor in ["lattice", "stream"]:
                camelot_tables = camelot.read_pdf(pdf_path, flavor=flavor, pages="all")
                for t in camelot_tables:
                    df = t.df
                    if df is not None and df.shape[0] > 0 and df.shape[1] > 0:
                        tables.append(df)
                if len(tables) > 0:
                    return tables
        except Exception:
            pass

    if "tabula" in EXTRACT_BACKENDS:
        try:
            dfs = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)
            for df in dfs:
                if df is not None and df.shape[0] > 0 and df.shape[1] > 0:
                    tables.append(df)
            if len(tables) > 0:
                return tables
        except Exception:
            pass

    if "pdfplumber" in EXTRACT_BACKENDS:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    if max_pages and page_idx >= max_pages:
                        break
                    page_tables = page.extract_tables()
                    for tbl in page_tables:
                        df = pd.DataFrame(tbl)
                        if df.shape[0] > 0 and df.shape[1] > 0:
                            tables.append(df)
            if len(tables) > 0:
                return tables
        except Exception:
            pass

    return tables


def clean_and_unify_tables(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Unify multiple extracted tables into one:
    - Promote first row to header if header-like
    - Remove empty rows/cols
    - Trim whitespace
    - Align columns across tables via union
    """
    cleaned = []
    for df in dfs:
        df = df.copy()
        first_row = df.iloc[0].astype(str).tolist()
        header_like = len(set(first_row)) == len(first_row)
        if header_like:
            df.columns = [str(c).strip() for c in first_row]
            df = df.iloc[1:].reset_index(drop=True)
        else:
            df.columns = [f"col_{i}" for i in range(df.shape[1])]

        df = df.applymap(lambda x: str(x).strip() if pd.notna(x) else x)
        df = df.dropna(how="all")
        df = df.loc[:, ~df.columns.duplicated()]
        df = df[[c for c in df.columns if not all(df[c].astype(str).str.strip().isin(["", "None", "nan"]))]]

        if df.shape[0] > 0 and df.shape[1] > 0:
            cleaned.append(df)

    if not cleaned:
        raise ValueError("No usable tables extracted from the PDF.")

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
# Typing & Text conversion
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


def row_to_text(row: pd.Series) -> str:
    parts = []
    for col, val in row.items():
        if pd.isna(val):
            continue
        s = str(val).strip()
        if s:
            parts.append(f"{col}: {s}")
    return "; ".join(parts)


# -----------------------
# Gemini setup & calls
# -----------------------
def setup_gemini(api_key: Optional[str] = None):
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Gemini API key not found. Set GEMINI_API_KEY env or pass --api_key.")
    genai.configure(api_key=key)


def gemini_embed_texts(texts: List[str], embed_model: str = "text-embedding-004") -> np.ndarray:
    """
    Use Gemini embeddings API to encode a list of strings.
    Returns a normalized numpy array of shape (N, D).
    """
    # Batch embedding: keep batch sizes moderate
    embeddings = []
    batch_size = 128
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = genai.embed_content(model=embed_model, content=batch)
        # resp['embedding'] when single item; for batch, resp['embeddings']
        vecs = resp.get("embeddings") or [resp["embedding"]]
        embeddings.extend([np.array(v["values"], dtype=np.float32) for v in vecs])
    arr = np.vstack(embeddings)
    # Normalize
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms
    return arr


def gemini_generate_summary(df: pd.DataFrame,
                            numeric_cols: List[str],
                            categorical_cols: List[str],
                            text_cols: List[str],
                            model_name: str = "gemini-1.5-flash") -> str:
    """
    Ask Gemini to produce a concise analytical narrative:
    - Summarize dataset shape and key columns
    - Describe distributions and potential insights
    - Suggest plots or clusters found
    """
    model = genai.GenerativeModel(model_name)
    # Create a compact JSON snapshot to avoid sending huge data
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
def reduce_dimensions(embeddings: np.ndarray, method: str = "umap", random_state: int = 42) -> Tuple[np.ndarray, str]:
    if method == "umap" and HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=random_state)
        coords = reducer.fit_transform(embeddings)
        label = "UMAP"
    else:
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(embeddings)
        label = "PCA"
    return coords, label


def fig_to_base64_png(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_scatter_plot(df: pd.DataFrame, coords: np.ndarray, color_col: Optional[str]) -> str:
    plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    title = "Embedding scatter (2D)"
    fig, ax = plt.subplots(figsize=(8, 6))
    if color_col and color_col in df.columns:
        plot_df[color_col] = df[color_col].astype(str)
        sns.scatterplot(data=plot_df, x="x", y="y", hue=color_col, s=35, ax=ax, palette="tab10")
        ax.legend(loc="best", title=color_col, fontsize=8)
        title += f" colored by '{color_col}'"
    else:
        sns.scatterplot(data=plot_df, x="x", y="y", s=35, ax=ax, color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
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
  <p class="note">This report summarizes patterns in the table and shows a 2D projection of semantic similarity between records. It is informational and does not constitute medical advice.</p>
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
def main(pdf_path: str, output_dir: str, api_key: Optional[str], color_by: Optional[str],
         summary_model: str, embed_model: str):
    os.makedirs(output_dir, exist_ok=True)
    setup_gemini(api_key)

    print(f"Extracting tables from: {pdf_path}")
    dfs = extract_tables(pdf_path)
    if not dfs:
        raise RuntimeError("No tables found. Ensure the PDF contains actual tables, not images of tables.")

    print(f"Found {len(dfs)} table(s). Cleaning and unifying...")
    df = clean_and_unify_tables(dfs)
    df = df.dropna(how="all").reset_index(drop=True)

    print("Detecting column types...")
    numeric_cols, categorical_cols, text_cols = detect_column_types(df)
    print(f"Numeric: {numeric_cols}")
    print(f"Categorical: {categorical_cols}")
    print(f"Text: {text_cols}")

    print("Preparing row texts for embedding...")
    texts = [row_to_text(df.iloc[i]) for i in range(df.shape[0])]

    print("Embedding rows with Gemini...")
    embeddings = gemini_embed_texts(texts, embed_model=embed_model)

    print("Building FAISS index and storing metadata...")
    index = build_faiss_index(embeddings)
    metadata = df.to_dict(orient="records")

    faiss_path = os.path.join(output_dir, "faiss.index")
    meta_path = os.path.join(output_dir, "metadata.json")
    faiss.write_index(index, faiss_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Reducing dimensions for visualization...")
    coords, method_label = reduce_dimensions(embeddings, method="umap" if HAS_UMAP else "pca")
    chosen_color = choose_color_column(df, categorical_cols, color_by)

    print("Building scatter plot...")
    scatter_png_b64 = build_scatter_plot(df, coords, chosen_color)

    print("Building interactive table preview...")
    table_html_fragment = build_table_plotly_html(df)

    print("Generating narrative analysis with Gemini...")
    narrative = gemini_generate_summary(df, numeric_cols, categorical_cols, text_cols, model_name=summary_model)

    print("Composing HTML report...")
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
    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_html)

    df.to_csv(os.path.join(output_dir, "cleaned_table.csv"), index=False)

    print("Done.")
    print(f"Saved:\n- {report_path}\n- {faiss_path}\n- {meta_path}\n- {os.path.join(output_dir, 'cleaned_table.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF → Gemini → Vector DB → Visual Report")
    parser.add_argument("--pdf_path", type=str, required=True, help="Path to the input PDF")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--color_by", type=str, default=None, help="Column to color the scatter plot")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API key (or set GEMINI_API_KEY env)")
    parser.add_argument("--summary_model", type=str, default="gemini-1.5-flash", help="Model for narrative")
    parser.add_argument("--embed_model", type=str, default="text-embedding-004", help="Embedding model")
    args = parser.parse_args()

    main(args.pdf_path, args.output_dir, args.api_key, args.color_by, args.summary_model, args.embed_model)
