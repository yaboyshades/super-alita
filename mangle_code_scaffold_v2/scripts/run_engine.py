import json
import sqlite3
import sys

RULES = {}

RULES["untested_function"] = """
SELECT s.symbol, s.file, c.score AS complexity
FROM symbol s
JOIN complexity c ON c.symbol = s.symbol
WHERE s.kind = 'function'
  AND c.score >= 0.3
  AND s.symbol NOT IN (SELECT target_sym FROM tests_targets)
"""

RULES["orphan_complex"] = """
WITH indeg AS (
  SELECT callee AS sym, COUNT(*) AS d FROM calls GROUP BY callee
), outdeg AS (
  SELECT caller AS sym, COUNT(*) AS d FROM calls GROUP BY caller
)
SELECT s.symbol, s.file, c.score
FROM symbol s
JOIN complexity c ON c.symbol = s.symbol
LEFT JOIN indeg i ON i.sym = s.symbol
LEFT JOIN outdeg o ON o.sym = s.symbol
WHERE s.kind='function'
  AND c.score >= 0.6
  AND IFNULL(i.d,0)=0 AND IFNULL(o.d,0)=0
"""

RULES["cycle"] = """
SELECT a.fileA, a.fileB
FROM dep a
JOIN dep b ON a.fileA = b.fileB AND a.fileB = b.fileA
WHERE a.fileA <> a.fileB
"""

RULES["hot_path"] = """
WITH indeg AS (
  SELECT callee AS sym, COUNT(*) AS d FROM calls GROUP BY callee
)
SELECT s.symbol, s.file, c.score AS complexity, IFNULL(i.d,0) AS indegree
FROM symbol s
JOIN complexity c ON c.symbol = s.symbol
LEFT JOIN indeg i ON i.sym = s.symbol
WHERE s.kind='function'
  AND c.score >= 0.5
  AND s.symbol NOT IN (SELECT target_sym FROM tests_targets)
  AND IFNULL(i.d,0) >= 1
"""

RULES["reinvention_json"] = """
SELECT s.symbol, s.file
FROM symbol s
WHERE s.kind='function'
  AND (LOWER(s.symbol) LIKE '%::to_json%' OR LOWER(s.symbol) LIKE '%::json_%')
  AND NOT EXISTS (
    SELECT 1 FROM imports i WHERE i.file = s.file AND i.module = 'json'
  )
"""

def run_rules(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    out = {"db": db_path, "rules": {}, "summary": {}}
    for name, sql in RULES.items():
        rows = conn.execute(sql).fetchall()
        out["rules"][name] = [dict(r) for r in rows]
        out["summary"][name] = len(rows)
    return out

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_engine.py <facts_db> <out_json>")
        sys.exit(1)
    db, out_json = sys.argv[1], sys.argv[2]
    result = run_rules(db)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote report to {out_json}")
