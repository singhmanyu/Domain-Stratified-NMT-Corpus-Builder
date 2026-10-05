"""Adds a UTF-8 BOM to csv files that lack one.

Without it, Excel on Windows reads a csv as the system ANSI codepage and
every Devanagari character comes out as mojibake. Pandas doesn't care either
way - we read with utf-8-sig everywhere - so this is purely so the files open
correctly by double-click.

Byte-level prepend, so a 600MB file takes seconds rather than a full pandas
reparse.

    python fix_csv_bom.py
"""

import glob
import os
import shutil

import config as cfg

BOM = b"\xef\xbb\xbf"


def has_bom(path):
    with open(path, "rb") as f:
        return f.read(3) == BOM


def add_bom(path):
    tmp = path + ".tmp"
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        dst.write(BOM)
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    os.replace(tmp, path)


def main():
    paths = (glob.glob(os.path.join(cfg.DATA_DIR, "*.csv"))
             + glob.glob(os.path.join(cfg.SPLIT_DIR, "*.csv")))
    if not paths:
        print("no csv files found")
        return
    for p in sorted(paths):
        size = os.path.getsize(p) / 1e6
        if has_bom(p):
            print(f"  ok      {os.path.basename(p)}  ({size:.1f} MB)")
        else:
            add_bom(p)
            print(f"  FIXED   {os.path.basename(p)}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
