from collections import defaultdict

def is_permutation(str1, str2):
    if str1 is None or str2 is None:
        return False
    if len(str1) != len(str2):
        return False

    unique_char1 = defaultdict(int)
    unique_char2 = defaultdict(int)

    for char in str1:
        unique_char1[char] += 1
    for char in str1:
        unique_char2[char] += 1

    return unique_char1 == unique_char2

print(is_permutation('foo','ofo'))