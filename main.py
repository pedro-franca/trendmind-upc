import numpy as np
import duckdb

db_path = './data/predictions.duckdb'
con = duckdb.connect(db_path)
df = con.execute(f"SELECT * FROM predictions").df()
con.close()
print(df.head())
