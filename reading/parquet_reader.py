# This code pulls the size, data type, columns names, and sample rows of a parquet file
import pandas as pd

path = "/user/home/il22158/work/vEcoli/reading/results/growth_rate/growth_rate_timeseries_errored_round1_all.parquet"
df = pd.read_parquet(path)
print("Shape:", df.shape)
print("Dtypes:\n", df.dtypes)
print("Columns:", df.columns)
print("Sample rows:\n", df.head())
