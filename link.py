import fitz  # PyMuPDF
import camelot
import pandas as pd
import json
 
def find_table_link(pdf_path, keyword="Table 1"):
    """
    Searches for hyperlinks containing the keyword.
    Returns a list of matches with page and link info.
    """
    doc = fitz.open(pdf_path)
    matches = []
 
    for page_num in range(len(doc)):
        page = doc[page_num]
        links = page.get_links()
        for link in links:
            rect = fitz.Rect(link["from"])
            text_near_link = page.get_textbox(rect)
            if keyword.lower() in text_near_link.lower():
                matches.append({
                    "page": page_num + 1,
                    "text": text_near_link.strip(),
                    "link": link
                })
    doc.close()
    return matches
 
 
def extract_table_with_camelot(pdf_path, page_number):
    """
    Extracts tables from the given page using Camelot
    and returns them as a list of DataFrames.
    """
    tables = camelot.read_pdf(pdf_path, pages=str(page_number), flavor='lattice')
    if tables.n == 0:
        tables = camelot.read_pdf(pdf_path, pages=str(page_number), flavor='stream')
    return [t.df for t in tables]
 
 
def get_page_text(pdf_path, page_number):
    """
    Extracts all text from a given page number.
    """
    doc = fitz.open(pdf_path)
    text = doc[page_number - 1].get_text()
    doc.close()
    return text
 
 
if __name__ == "__main__":
    pdf_path = "Prot_000.pdf"  # <-- change to your file
    keyword = "Table 1"
    target_table_name = "Time and Events Schedule"
 
    # Step 1: Search for hyperlinks containing keyword
    found_links = find_table_link(pdf_path, keyword)
 
    if not found_links:
        print(f"No hyperlink found containing '{keyword}'.")
    else:
        for item in found_links:
            print(f"\nFound link text '{item['text']}' on page {item['page']}:")
            link_data = item["link"]
 
            # Step 2: If it’s an internal link, follow it
            if "page" in link_data and link_data["page"] is not None:
                dest_page = link_data["page"] + 1  # PyMuPDF is 0-indexed
                print(f"→ Link points to page {dest_page}")
 
                # Step 3: Check if page text contains the target table name
                page_text = get_page_text(pdf_path, dest_page)
                if target_table_name.lower() in page_text.lower():
                    print(f"✅ Found table titled '{target_table_name}' on page {dest_page}")
 
                    # Step 4: Extract tables using Camelot
                    dfs = extract_table_with_camelot(pdf_path, dest_page)
 
                    if dfs:
                        df = dfs[0]  # take the first table
                        print("\n--- Extracted DataFrame ---")
                        print(df)
                        df.to_pickle("my_dataframe.pkl")
 
                        # Step 5: Also provide JSON representation
                        json_data = df.to_json(orient="records")
                        print("\n--- Table Content (JSON) ---")
                        print(json_data)

                        json_str = json.dumps(json_data, indent=4)
                        with open("sample.json", "w") as f:
                            f.write(json_str)
 
                        # ✅ Stop after finding the correct table
                        break
                    else:
                        print(f"No tables detected on page {dest_page}.")
                else:
                    print(f"Page {dest_page} does not contain the title '{target_table_name}'.")
            elif "uri" in link_data and link_data["uri"]:
                print(f"→ External link found: {link_data['uri']}")