with open('./text/all_sentences.txt') as f:
    full_corpus = f.read().splitlines()

NP = full_corpus[:12]
NPVP = full_corpus[12:120]
SOV = full_corpus[120:]


__all__ = ["NP", "NPVP", "SOV"]