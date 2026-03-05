import re

import numpy as np

ALPHABET = list("abcdefghijklmnopqrstuvwxyz")
VOWELS = list("aeiou")
PAD_TOKEN = "[PAD]"

char_map = {v: i for i, v in enumerate(ALPHABET, start=1)}
char_map[PAD_TOKEN] = 0


def create_dataset(seq: str, stride: int = 5) -> list:
    dataset = []
    clean_seq = "".join(seq.split())
    seq_len = len(clean_seq)
    offset = stride // 2
    ptr = 0

    for i in range(seq_len):
        start_idx = max(i - offset, 0)
        end_idx = i + offset + 1
        pad_left = PAD_TOKEN * max(offset - i, 0)
        pad_right = PAD_TOKEN * max(end_idx - seq_len, 0)

        substr = pad_left + clean_seq[start_idx:end_idx] + pad_right
        while ptr < len(seq) and seq[ptr] == " ":
            ptr += 1

        target = 0
        if ptr + 1 < len(seq) and seq[ptr + 1] == " ":
            target = 1

        ptr += 1
        dataset.append((substr, target))
    return dataset


def transform_to_tensor(seq: str):
    vectors = []
    tokens = re.findall(r"\[PAD\]|[a-z]", seq)

    for char in tokens:
        char_vec = np.zeros(28)  # 27 for ID + 1 for Vowel
        idx = char_map[char]
        char_vec[idx] = 1
        char_vec[27] = 1.0 if char in VOWELS else 0.0
        vectors.extend(char_vec)

    return np.array(vectors)


if __name__ == "__main__":
    print(create_dataset("i like ai"))
