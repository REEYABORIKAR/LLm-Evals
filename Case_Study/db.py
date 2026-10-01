import sqlite3
import pandas as pd

MATCHES_CSV = "data/matches.csv"
DELIVERIES_CSV = "data/deliveries.csv"
DB_PATH = "ipl_2021_2024.db"
YEARS = {2021, 2022, 2023, 2024}

matches = pd.read_csv(MATCHES_CSV)
matches["date"] = pd.to_datetime(matches["date"], errors="coerce", dayfirst=False)

if matches["date"].isna().any():
    fallback = pd.to_datetime(
        matches.loc[matches["date"].isna(), "date"] if False else
        pd.read_csv(MATCHES_CSV)["date"],
        errors="coerce", dayfirst=True
    )
    matches["date"] = matches["date"].fillna(fallback)

matches["year"] = matches["date"].dt.year

matches_filtered = matches[matches["year"].isin(YEARS)].copy()
match_ids = set(matches_filtered["id"].unique())
print(f"Matches in 2021-2024: {len(matches_filtered)}")

deliveries = pd.read_csv(DELIVERIES_CSV)
print("Delivery columns:")
print(deliveries.columns.tolist())

deliveries_filtered = deliveries[deliveries["match_ids"].isin(match_ids)].copy()
print(f"Deliveries for those matches: {len(deliveries_filtered)}")

conn = sqlite3.connect(DB_PATH)

matches_to_save = matches_filtered.drop(columns=["year"])

matches_to_save.to_sql("matches", conn, if_exists="replace", index=False)
deliveries_filtered.to_sql("deliveries", conn, if_exists="replace", index=False)

cur = conn.cursor()
cur.execute("CREATE INDEX IF NOT EXISTS idx_deliveries_match ON deliveries(match_ids)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_matches_id ON matches(id)")
conn.commit()

print("\n--- Verification ---")
print("matches rows:", cur.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
print("deliveries rows:", cur.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0])

orphans = cur.execute("""
    SELECT COUNT(*) FROM deliveries d
    LEFT JOIN matches m ON d.match_ids = m.id
    WHERE m.id IS NULL
""").fetchone()[0]
print("orphan deliveries (should be 0):", orphans)

yr = cur.execute("""
    SELECT MIN(date), MAX(date) FROM matches
""").fetchone()
print("date range:", yr)

conn.close()
print(f"\nDone. Wrote {DB_PATH}")