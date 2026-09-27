from collections import defaultdict, Counter

def get_vocab(corpus: list[str]) -> dict[str, int]:
    """Split corpus words into character sequences with end-of-word marker."""
    vocab = defaultdict(int)
    for word in corpus:
        # Represent word as tuple of chars + end-of-word marker
        token = tuple(list(word) + ['_'])
        vocab[token] += 1
    return dict(vocab)

def get_pair_counts(vocab: dict[tuple, int]) -> Counter:
    """Count frequency of every adjacent symbol pair across vocab."""
    pairs = Counter()
    for token_tuple, freq in vocab.items():
        for i in range(len(token_tuple) - 1):
            pairs[(token_tuple[i], token_tuple[i + 1])] += freq
    return pairs

def merge_pair(pair: tuple[str, str], vocab: dict[tuple, int]) -> dict[tuple, int]:
    """Merge all instances of the given pair in the vocab."""
    new_vocab = {}
    merged = ''.join(pair)
    for token_tuple, freq in vocab.items():
        new_token = []
        i = 0
        while i < len(token_tuple):
            # If current and next symbol match the pair, merge them
            if i < len(token_tuple) - 1 and token_tuple[i] == pair[0] and token_tuple[i + 1] == pair[1]:
                new_token.append(merged)
                i += 2
            else:
                new_token.append(token_tuple[i])
                i += 1
        new_vocab[tuple(new_token)] = freq
    return new_vocab

def train_bpe(corpus: list[str], num_merges: int) -> list[tuple[str, str]]:
    """Train BPE and return the learned merge rules in order."""
    vocab = get_vocab(corpus)
    merge_rules = []

    for _ in range(num_merges):
        pairs = get_pair_counts(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_pair(best_pair, vocab)
        merge_rules.append(best_pair)
        print(f"Merged: {best_pair} -> {''.join(best_pair)}")

    return merge_rules

def encode(word: str, merge_rules: list[tuple[str, str]]) -> list[str]:
    """Apply learned merge rules to encode a new word."""
    tokens = list(word) + ['_']
    for pair in merge_rules:
        merged = ''.join(pair)
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens

# Example
corpus = [
    "low", "low", "low", "low", "low",
    "lower", "lower",
    "newest", "newest", "newest", "newest", "newest", "newest",
    "widest", "widest", "widest"
]

merge_rules = train_bpe(corpus, num_merges=10)

print("\nEncoding 'lowest':", encode("lowest", merge_rules))
print("Encoding 'newer':", encode("newer", merge_rules))