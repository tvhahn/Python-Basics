def unique_chars(string):
    if string is None:
        return False

    char_set = set()
    for char in string:
        if char in char_set:
            return False
        else:
            char_set.add(char)
    return True

print(unique_chars('bar'))