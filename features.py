from collections import Counter

import numpy as np

VOWELS = [ord(c) for c in "AEIOU"]


def duplicate_items(l):
    return [x for x, count in Counter(l).items() if count > 1]


def duplicate_items_order(l):
    total = 0
    for i in range(len(l) - 1):
        if l[i] == l[i + 1]:
            total += i
    return total


def alphab(c):
    i = c - ord("A")

    # If we have non text characters, treat as A
    if i > 25:
        return 0

    return i


def name_features(name):
    # 0 letter frequency
    # 26 letter order
    # 52 length
    # 53 vowel percentage
    # 54 final character
    # 55 final character - 1
    # 56 final character - 2
    # 57 double letter counts
    # 58 double letter order
    arr = np.zeros(26 + 26 + 7)

    # Letter frequencies and orders
    for i, c in enumerate(name):
        arr[alphab(c)] += 1
        arr[alphab(c) + 26] += i + 1

    # Length
    arr[52] = len(name)

    # Vowel percentage
    count_vowels = len([c for c in name if c in VOWELS])
    arr[53] = int((count_vowels / len(name)) * 100)

    # Last character
    arr[54] = name[-1]

    # Last but one character
    arr[55] = name[-2]

    # Last but two character
    arr[56] = name[-3]

    # Double character counts and order
    arr[57] = len(duplicate_items(name))
    arr[58] = duplicate_items_order(name)

    return arr


name_features_map = np.vectorize(name_features, otypes=[np.ndarray])


def featurize(names):
    return np.array(name_features_map(names).tolist())
