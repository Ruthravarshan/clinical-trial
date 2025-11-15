import os
import fitz  # PyMuPDF
import google.generativeai as genai
import faiss
import numpy as np
import pdfplumber
import pandas as pd
from dotenv import load_dotenv

# ---------------- Configuration ----------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

EMBED_MODEL = "text-embedding-004"
QUERY = "Time and Events Schedule"  # anchor keyword
PDF_PATH = "Prot_000.pdf"           # <-- hard-coded PDF path here

MIN_HEADER_JACCARD = 0.2  # header similarity threshold

# ---------------- Helpers ----------------
def extract_pages_text(pdf_path: str):
    """Extract plain text from every page using PyMuPDF."""
    texts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            txt = page.get_text("text") or ""
            if not txt.strip():
                txt = "empty page"
            texts.append(txt)
    return texts

def _to_numpy_embedding(resp):
    emb = resp.get("embedding")
    values = emb["values"] if isinstance(emb, dict) and "values" in emb else emb
    return np.array(values, dtype="float32")

def embed_text(text: str, task_type: str):
    resp = genai.embed_content(model=EMBED_MODEL, content=text, task_type=task_type)
    return _to_numpy_embedding(resp)

def embed_pages(pages: list):
    vecs = []
    for p in pages:
        vecs.append(embed_text(p, task_type="retrieval_document"))
    return np.vstack(vecs).astype("float32")

def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def retrieve_anchor_page(index, query: str):
    q = embed_text(query, task_type="retrieval_query").reshape(1, -1).astype("float32")
    D, I = index.search(q, k=1)
    return int(I[0][0]) + 1  # 1-based page number

# ---------------- Table extraction ----------------
def extract_first_table(pdf_path: str, page_number: int):
    """Extract the first table from a page using pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        tables = page.extract_tables()
        if not tables:
            return None
        return pd.DataFrame(tables[0])

def table_signature(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    num_cols = df.shape[1]
    header_row = None
    for i in range(len(df)):
        row = df.iloc[i]
        if any(str(x).strip() for x in row):
            header_row = row
            break
    if header_row is None:
        header_row = df.iloc[0]
    header_tokens = tuple(str(x).strip().lower() for x in header_row.fillna(""))
    return (num_cols, header_tokens)

def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def is_continuation(anchor_sig, candidate_sig, min_header_jaccard=MIN_HEADER_JACCARD):
    if anchor_sig is None or candidate_sig is None:
        return False
    if anchor_sig[0] != candidate_sig[0]:
        return False
    a_hdr = [h for h in anchor_sig[1] if h]
    c_hdr = [h for h in candidate_sig[1] if h]
    if not a_hdr or not c_hdr:
        return True
    return jaccard(a_hdr, c_hdr) >= min_header_jaccard

# ---------------- Main Logic ----------------
def get_schedule_pages_and_table(pdf_path: str, query: str = QUERY):
    pages_text = extract_pages_text(pdf_path)
    embeddings = embed_pages(pages_text)
    index = build_faiss_index(embeddings)
    anchor_page = retrieve_anchor_page(index, query=query)

    anchor_df = extract_first_table(pdf_path, anchor_page)
    anchor_sig = table_signature(anchor_df)

    schedule_pages = [anchor_page]
    tables = []
    if anchor_df is not None:
        tables.append(anchor_df)

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)

    # Scan forward
    for p in range(anchor_page + 1, total_pages + 1):
        df = extract_first_table(pdf_path, p)
        sig = table_signature(df)
        if is_continuation(anchor_sig, sig):
            schedule_pages.append(p)
            if df is not None:
                tables.append(df)
        else:
            break

    merged_df = pd.concat(tables, ignore_index=True) if tables else None
    return schedule_pages, merged_df

# ---------------- Run ----------------
if __name__ == "__main__":
    try:
        pages, merged_df = get_schedule_pages_and_table(PDF_PATH, query=QUERY)
        print(f"Schedule table found on pages: {pages}")
        if merged_df is not None:
            print("\nMerged table preview:")
            print(merged_df.head())
            merged_df.to_csv("schedule_table.csv", index=False, encoding="utf-8-sig")
            print("\nSaved merged table as schedule_table.csv")
        else:
            print("\nNo tables extracted from the detected pages.")
    except Exception as e:
        print(f"Error: {e}")
        print("Hint: Ensure GEMINI_API_KEY is set and dependencies installed: ")
        print("  pip install google-generativeai faiss-cpu pymupdf pdfplumber pandas numpy")
