import camelot
import pdfplumber
import pandas as pd
from rapidfuzz import fuzz
import os

# ---------- CONFIGURATION ----------
PDF_PATH = "Prot_2.pdf"  # Replace with your PDF path
OUTPUT_DIR = "output_tables"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keywords to identify candidate pages
KEYWORDS = [
    "time and events",
    "schedule of assessments",
    "schedule of visits",
    "schedule of events",
    "assessment schedule",
    "visit schedule",
    "study schedule"
]

# Extra header keywords that must appear in the table header row
TABLE_HEADER_KEYWORDS = [
    "trial activity",
    "run-in",
    "screening period",
    "treatment period",
    "follow-up period",
    "week",
    "day",
    "visits"
]


# ---------- STEP 1: DETECT CANDIDATE PAGES ----------
def detect_candidate_pages(pdf_path, keywords=KEYWORDS, threshold=75):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            for line in text.split("\n"):
                for kw in keywords:
                    score = fuzz.partial_ratio(line.lower(), kw)
                    if score >= threshold:
                        pages.append(i + 1)
                        break
    pages = sorted(list(set(pages)))
    print(f"Candidate pages (possible table pages): {pages}")
    return pages


# ---------- STEP 2: EXTRACT TABLES ----------
def extract_tables(pdf_path, pages):
    all_tables = []
    if not pages:
        print("No candidate pages detected.")
        return all_tables

    page_range = ",".join(str(p) for p in pages)
    print(f"Extracting tables from pages: {page_range}")

    try:
        tables = camelot.read_pdf(pdf_path, pages=page_range, flavor="lattice", strip_text="\n")
        for t in tables:
            all_tables.append((t.page, t.df))
    except Exception as e:
        print("Camelot lattice failed, trying stream mode:", e)
        for p in pages:
            try:
                tables = camelot.read_pdf(pdf_path, pages=str(p), flavor="stream", strip_text="\n")
                for t in tables:
                    all_tables.append((t.page, t.df))
            except Exception as inner_e:
                print(f"Stream mode failed on page {p}: {inner_e}")
                continue

    print(f"Found {len(all_tables)} tables across {len(pages)} candidate pages.")
    return all_tables


# ---------- STEP 2b: FILTER TABLES BY HEADER CONTENT ----------
def filter_schedule_tables(tables):
    filtered = []
    for page, df in tables:
        if df.empty:
            continue
        header_text = " ".join(df.iloc[0].astype(str).tolist()).lower()
        matches = sum(1 for kw in TABLE_HEADER_KEYWORDS if kw in header_text)
        if matches >= 3:  # require at least 3 key header terms
            print(f"Keeping table on page {page} (matched {matches} header keywords)")
            filtered.append((page, df))
        else:
            print(f"Skipping table on page {page} (header={header_text})")
    return filtered


# ---------- STEP 3: MERGE MULTI-PAGE TABLES ----------
def merge_tables(tables):
    merged_tables = []
    used = set()

    for i, (page, df) in enumerate(tables):
        if page in used:
            continue

        group = [df]
        pages = [page]
        used.add(page)

        # look ahead for continuation
        for j in range(i + 1, len(tables)):
            next_page, next_df = tables[j]
            if next_page in used:
                continue
            try:
                h1 = " ".join(df.iloc[0].astype(str).tolist())
                h2 = " ".join(next_df.iloc[0].astype(str).tolist())
                score = fuzz.token_sort_ratio(h1, h2)
                if score > 70:
                    print(f"Merging page {page} + {next_page} (header sim={score:.1f})")
                    group.append(next_df)
                    pages.append(next_page)
                    used.add(next_page)
                else:
                    break
            except Exception:
                continue

        merged_df = pd.concat(group, ignore_index=True)
        merged_tables.append((pages, merged_df))

    print(f"Merged into {len(merged_tables)} combined tables.")
    return merged_tables


# ---------- STEP 4: SAVE OUTPUT AS PICKLE ----------
def save_tables_as_pickle(merged_tables):
    saved_paths = []
    for idx, (pages, df) in enumerate(merged_tables, start=1):
        out_path = os.path.join(
            OUTPUT_DIR,
            f"schedule_table_{idx}_pages_{'_'.join(map(str, pages))}.pkl"
        )
        df.to_pickle(out_path)
        saved_paths.append(out_path)
        print(f"Saved schedule table #{idx} from pages {pages} -> {out_path}")
    return saved_paths


# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    print("Starting schedule table extraction pipeline...\n")
    candidate_pages = detect_candidate_pages(PDF_PATH)
    tables = extract_tables(PDF_PATH, candidate_pages)
    schedule_tables = filter_schedule_tables(tables)
    merged = merge_tables(schedule_tables)
    saved = save_tables_as_pickle(merged)

    if merged:
        pages, df = merged[0]
        print("\nPreview of extracted schedule table:")
        print(df.head(20))
    else:
        print("No schedule tables found. Check if the PDF is scanned or headers differ.")
