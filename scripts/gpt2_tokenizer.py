
import json
import re
from pathlib import Path

def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

def load_encoder(bpe_dir: Path):
    vocab_json = bpe_dir / "vocab.json"
    encoder_json = bpe_dir / "encoder.json"
    if vocab_json.exists():
        encoder = json.loads(vocab_json.read_text())
    elif encoder_json.exists():
        encoder = json.loads(encoder_json.read_text())
    else:
        raise FileNotFoundError(f"Missing vocab.json or encoder.json in {bpe_dir}")
    return {int(v): k for k, v in encoder.items()}

def build_bpe(vocab_path: Path, merges_path: Path):
    encoder = json.loads(vocab_path.read_text())
    merges = merges_path.read_text().splitlines()[1:]
    bpe_ranks = {tuple(m.split()): i for i, m in enumerate(merges)}
    byte_encoder = bytes_to_unicode()
    cache = {}

    def get_pairs(word):
        pairs = set()
        prev_char = word[0]
        for ch in word[1:]:
            pairs.add((prev_char, ch))
            prev_char = ch
        return pairs

    def bpe(token):
        if token in cache:
            return cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            cache[token] = token
            return token
        while True:
            bigram = min(pairs, key=lambda p: bpe_ranks.get(p, float("inf")))
            if bigram not in bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        out = " ".join(word)
        cache[token] = out
        return out

    pat = re.compile(
        r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
    )

    def encode(text):
        out = []
        for token in pat.findall(text):
            token_bytes = token.encode("utf-8")
            token_trans = "".join(byte_encoder[b] for b in token_bytes)
            out.extend(encoder[t] for t in bpe(token_trans).split(" "))
        return out

    return encode

def build_bpe_from_dir(bpe_dir: Path):
    vocab = bpe_dir / "vocab.json"
    if not vocab.exists():
        vocab = bpe_dir / "encoder.json"
    if not vocab.exists():
        raise FileNotFoundError(f"Missing vocab.json or encoder.json in {bpe_dir}")
    merges = bpe_dir / "merges.txt"
    if not merges.exists():
        merges = bpe_dir / "vocab.bpe"
    if not merges.exists():
        raise FileNotFoundError(f"Missing merges.txt or vocab.bpe in {bpe_dir}")
    return build_bpe(vocab, merges)

def decode_tokens(tokens, decoder):
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    text = "".join(decoder.get(t, "") for t in tokens)
    byte_arr = bytearray(byte_decoder[c] for c in text)
    return byte_arr.decode("utf-8", errors="replace")
