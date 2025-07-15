"""Utility module to load Brocanto2 grammar constructions."""

from pathlib import Path

CORPUS_PATH = Path(__file__).parent / "text" / "all_sentences.txt"

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    full_corpus = f.read().splitlines()

NP = full_corpus[:12]
NPVP = full_corpus[12:120]
SOV = full_corpus[120:]

# __all__ 使得 import * 时只导出这三个变量
__all__ = ["NP", "NPVP", "SOV"]
