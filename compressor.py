import pandas as pd
import time

print("⏳ Loading massive 1.5 GB CSV into memory... (This might take a minute)")
start_time = time.time()

# Load your huge original CSV
df = pd.read_csv("MASTER_5M_AuditData.csv")

print(f"✅ CSV Loaded! Took {round(time.time() - start_time, 2)} seconds.")
print(f"📊 Total Rows: {len(df):,}")
print("🗜️ Compressing into ultra-fast Parquet format...")

# Save it as a highly compressed Parquet file using the pyarrow engine
df.to_parquet("MASTER_5M_AuditData.parquet", engine="pyarrow", index=False)

print("🚀 DONE! Your new compressed file 'MASTER_5M_AuditData.parquet' is ready!")