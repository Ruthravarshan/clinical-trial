import os
import fitz 
import google.generativeai as genai
import faiss
import numpy as np
import camelot
import pandas as pd


# Configuration
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyD8HQXLPT9JoMORrYfil1voZCIj6_G6nyA")
genai.configure(api_key=API_KEY)

EMBED_MODEL = "text-embedding-004"


# Helpers
def extract_pages(pdf_path: str) -> list:
    """Extract text content from each page of the PDF."""
    texts = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            # "text" layout gives plain text; ensure non-empty string
            txt = page.get_text("text") or ""
            # Avoid empty strings which can error the embedding call
            if not txt.strip():
                txt = "empty page"
            texts.append(txt)
    return texts

def _to_numpy_embedding(resp) -> np.ndarray:
    """Convert an embed_content response to a float32 numpy vector."""
    emb = resp.get("embedding")
    # Handle both dict with 'values' or direct list of floats
    if isinstance(emb, dict) and "values" in emb:
        values = emb["values"]
    else:
        values = emb
    return np.array(values, dtype="float32")

def embed_text(text: str, task_type: str = "retrieval_document") -> np.ndarray:
    """Embed a single text string."""
    resp = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type=task_type
    )
    return _to_numpy_embedding(resp)

def embed_pages(pages: list) -> np.ndarray:
    """Embed each page and return a 2D numpy array [num_pages, dim] in float32."""
    vectors = []
    for page_text in pages:
        vec = embed_text(page_text, task_type="retrieval_document")
        vectors.append(vec)
    return np.vstack(vectors).astype("float32")

def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build an L2 FAISS index over the embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def find_schedule_page(index: faiss.IndexFlatL2, query: str, k: int = 1) -> int:
    """Return the best matching 1-indexed page number for the query."""
    query_vec = embed_text(query, task_type="retrieval_query").reshape(1, -1).astype("float32")
    D, I = index.search(query_vec, k=k)
    # Top-1 match; FAISS indices are 0-based, PDF page numbers are typically 1-based
    return int(I[0][0]) + 1

def get_schedule_page(pdf_path: str, query: str = "Time and Events Schedule") -> int:
    """End-to-end: extract pages, embed, build index, and find the best matching page."""
    pages = extract_pages(pdf_path)
    if len(pages) == 0:
        raise ValueError("No pages extracted from the PDF.")
    embeddings = embed_pages(pages)
    index = build_index(embeddings)
    page_number = find_schedule_page(index, query=query, k=1)
    return page_number

#table vis
def extract_table_from_page(pdf_path, page_number):
    """
    Extracts tables from the specified page using Camelot.
    Returns the first table as a DataFrame, or None if no tables found.
    """
    # Try lattice mode first
    tables = camelot.read_pdf(pdf_path, pages=str(page_number), flavor='lattice')
    
    # If no tables found, try stream mode
    if tables.n == 0:
        tables = camelot.read_pdf(pdf_path, pages=str(page_number), flavor='stream')
    
    if tables.n > 0:
        df = tables[0].df
        return df
    else:
        return None

# Run
if __name__ == "__main__":
    pdf_path = "Prot_000.pdf"
    try:
        page_num = get_schedule_page(pdf_path, query="Time and Events Schedule") +1
        print(f"Table found on page: {page_num}")
    except Exception as e:
        # Common cause: invalid API key or network issues
        print(f"Error: {e}")
        print("Hint: Ensure your Gemini API key is valid and set in GEMINI_API_KEY, "
              "and that the 'google-generativeai' library is up to date (pip install --upgrade google-generativeai).")
        
    df = extract_table_from_page(pdf_path, page_num)
    if df is not None:
        print("\n Table extracted successfully:")
        print(df)

        # Save for further processing
        df.to_pickle("my_dataframe.pkl")

    else:
        print(f"\n No tables found on page {page_num}.")
