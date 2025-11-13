import camelot
import pandas as pd

# Path to your PDF
pdf_path = "Prot_000.pdf"

# Extract all tables from all pages
tables = camelot.read_pdf(pdf_path, pages='44,45', flavor='lattice')

print(f"Total tables found: {len(tables)}")

# Combine all tables into one DataFrame
dfs = [table.df for table in tables]

# Optionally inspect the first few tables
for i, df in enumerate(dfs):
    print(f"\n--- Table {i+1} ---")
    print(df.head())

# Merge all tables into a single structured dataframe (if needed)
combined_df = pd.concat(dfs, ignore_index=True)

# Clean column names
combined_df.columns = [col.strip() for col in combined_df.iloc[0]] # use first row as header
combined_df = combined_df[1:] # remove header row
combined_df.reset_index(drop=True, inplace=True)

# Save structured data
combined_df.to_csv("structured_clinical_trial_data.csv", index=False)
combined_df.to_excel("structured_clinical_trial_data.xlsx", index=False)

print("\n Structured table extracted and saved successfully!")