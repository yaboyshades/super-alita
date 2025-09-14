from src.globkit import analyze_file_collection, list_files

print("CSV files:", list_files("data/**/*.csv", recursive=True)[:5])
print(analyze_file_collection("data/**/*.{csv,json,xlsx}", recursive=True).head())
# print(load_csv_dataset("data/sales_2024_*.csv").shape)
