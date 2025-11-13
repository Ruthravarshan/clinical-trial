import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import json

# Step 1: Find the page linked by "Table 1"
def find_table1_link_page(pdf_path):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        links = page.get_links()
        for link in links:
            if link['kind'] == fitz.LINK_GOTO and 'Table 1' in page.get_text("text"):
                return link.get('page')
    return None

# Step 2: Extract table with title "Time and Events Schedule"
def extract_named_table(pdf_path, page_num, title_keyword="Time and Events Schedule"):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        text = page.extract_text()
        if title_keyword.lower() in text.lower():
            tables = page.extract_tables()
            for table in tables:
                if table and any(title_keyword.lower() in str(cell).lower() for row in table for cell in row if cell):
                    df = pd.DataFrame(table[1:], columns=table[0])
                    return df
    return None

# Step 3: Save outputs
def save_outputs(df, json_path, excel_path):
    df.to_json(json_path, orient='records', indent=2)
    df.to_excel(excel_path, index=False)
    print("✅ DataFrame:")
    print(df)

# Main execution
pdf_file = "Prot_000.pdf"  # Replace with your actual file path
linked_page = find_table1_link_page(pdf_file)

if linked_page is not None:
    df = extract_named_table(pdf_file, linked_page)
    if df is not None:
        save_outputs(df, "output.json", "output.xlsx")
    else:
        print("⚠️ Table with title 'Time and Events Schedule' not found on the linked page.")
else:
    print("⚠️ No hyperlink labeled 'Table 1' found.")
