import argparse
from pathlib import Path
import duckdb

parser = argparse.ArgumentParser(description="Prepare the Favorita competition training CSV.")
parser.add_argument("--train", type=Path, required=True)
args = parser.parse_args()
Path("work").mkdir(exist_ok=True)
if Path("work/favorita-panel.parquet").exists():
    raise FileExistsError("work/favorita-panel.parquet already exists")
db = duckdb.connect()
db.execute("SET memory_limit='6GB'")
db.execute("SET threads=6")
db.execute("""
COPY (
 SELECT store_nbr AS store, item_nbr AS product, CAST(date AS DATE) AS period,
        sum(unit_sales) AS sales, max(CAST(onpromotion AS INTEGER)) AS promotion
 FROM read_csv_auto(?, header=true)
 GROUP BY ALL ORDER BY store, product, period
) TO 'work/favorita-panel.parquet' (FORMAT PARQUET)
""", [str(args.train)])
print("Prepared work/favorita-panel.parquet")
