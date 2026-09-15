"""
Create a smaller .db3 bag by keeping every Nth frame.

Usage:
    python3 trim_bag.py for_hackathon/doubleT_platform          # every 5th frame (default)
    python3 trim_bag.py for_hackathon/doubleT_platform 3        # every 3rd frame
    python3 trim_bag.py for_hackathon/doubleT_platform 10       # every 10th frame
"""

import sqlite3
import sys
import os
import shutil


def trim_bag(bag_dir, step=5):
    db3_files = [f for f in os.listdir(bag_dir) if f.endswith('.db3')]
    if not db3_files:
        print(f"No .db3 files in {bag_dir}")
        return

    src_db = os.path.join(bag_dir, db3_files[0])
    out_dir = bag_dir + f"_trim{step}"
    os.makedirs(out_dir, exist_ok=True)

    # Copy metadata.yaml if exists
    meta = os.path.join(bag_dir, "metadata.yaml")
    if os.path.exists(meta):
        shutil.copy2(meta, out_dir)

    out_db = os.path.join(out_dir, db3_files[0])

    src = sqlite3.connect(src_db)
    src_cur = src.cursor()

    src_cur.execute("SELECT COUNT(*) FROM messages")
    total = src_cur.fetchone()[0]

    # Create destination with same schema
    dst = sqlite3.connect(out_db)
    dst_cur = dst.cursor()

    # Copy schema from source
    src_cur.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    for (sql,) in src_cur:
        if sql:
            dst_cur.execute(sql)

    # Copy every Nth message
    src_cur.execute("SELECT * FROM messages ORDER BY timestamp ASC")
    cols = [d[0] for d in src_cur.description]
    placeholders = ",".join(["?"] * len(cols))
    insert_sql = f"INSERT INTO messages ({','.join(cols)}) VALUES ({placeholders})"

    kept = 0
    for i, row in enumerate(src_cur):
        if i % step == 0:
            dst_cur.execute(insert_sql, row)
            kept += 1

    dst.commit()
    dst.close()
    src.close()

    src_size = os.path.getsize(src_db) / (1024 ** 3)
    dst_size = os.path.getsize(out_db) / (1024 ** 3)

    print(f"Original: {total} frames ({src_size:.1f} GB)")
    print(f"Trimmed:  {kept} frames ({dst_size:.1f} GB)")
    print(f"Saved to: {out_dir}/")
    print(f"\nRun:  python3 visualize_bag.py {out_dir}")


if __name__ == "__main__":
    bag_dir = sys.argv[1] if len(sys.argv) > 1 else "for_hackathon/doubleT_platform"
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    trim_bag(bag_dir, step)
