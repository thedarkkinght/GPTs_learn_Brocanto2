with open('D:/language acquisition/upload_scripts/upload_git/train_GPT/text/all_sentences.txt') as f:
    full_corpus = f.read().splitlines()

NP = full_corpus[:12]
NPVP = full_corpus[12:120]
SOV = full_corpus[120:]

# __all__ 使得 import * 时只导出这三个变量
__all__ = ["NP", "NPVP", "SOV"]