"""文档编码完整性核查（开发期工具）。"""
from __future__ import annotations

import pathlib

BASE = pathlib.Path("participant/P514")
BOM = b"\xef\xbb\xbf"
for n in ("REPORT.md", "LOG.md", "experiments.csv", "THIRD_PARTY.md", "submission.yaml"):
    p = BASE / n
    if not p.exists():
        print(f"{n:20s} 不存在")
        continue
    raw = p.read_bytes()
    try:
        s = raw.decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"{n:20s} 解码失败 {exc}")
        continue
    pua = sum(1 for c in s if 0xE000 <= ord(c) <= 0xF8FF)
    cjk = sum(1 for c in s if 0x4E00 <= ord(c) <= 0x9FFF)
    print(f"{n:20s} bytes={len(raw):6d} BOM={raw[:3] == BOM} PUA={pua:5d} CJK={cjk:5d}")
