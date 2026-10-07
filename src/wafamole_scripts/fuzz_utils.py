# fuzz_utils.py
# Vendored from WAF-A-MoLE (Demetrio et al., 2020), wafamole/payloadfuzzer/fuzz_utils.py

import re
import random
import string


# Replace one random occurrence of sub with wanted.
def replace_random(candidate, sub, wanted):
    occurrences = list(re.finditer(sub, candidate))
    if not occurrences:
        return candidate
    match = random.choice(occurrences)
    before = candidate[:match.start()]
    after = candidate[match.end():]
    return before + wanted + after


# Return which symbols from the dict actually appear in the payload.
def filter_candidates(symbols, payload):
    return [s for s in symbols.keys() if re.search(r'{}'.format(re.escape(s)), payload)]


# Return one random character, optionally including whitespace.
def random_char(spaces=True):
    chars = string.digits + string.ascii_letters + string.punctuation
    if spaces:
        chars += string.whitespace
    return random.choice(chars)


# Build a random string of random length up to max_len.
def random_string(max_len=5, spaces=True):
    return "".join([random_char(spaces=spaces) for i in range(random.randint(1, max_len))])


# Return a random true string comparison, e.g. 'x'='x'.
def string_tautology():
    value_s = random_string(random.randint(1, 5))
    tautologies = [
        "'{}'='{}'".format(value_s, value_s),
        "'{}' LIKE '{}'".format(value_s, value_s),
        "'{}'='{}'".format(value_s, value_s),
        "'{}' LIKE '{}'".format(value_s, value_s),
        "'{}'!='{}'".format(value_s, value_s + random_string(1, spaces=False)),
        "'{}'<>'{}'".format(value_s, value_s + random_string(1, spaces=False)),
        "'{}' NOT LIKE '{}'".format(value_s, value_s + random_string(1, spaces=False)),
        "'{}'!='{}'".format(value_s, value_s + random_string(1, spaces=False)),
        "'{}'<>'{}'".format(value_s, value_s + random_string(1, spaces=False)),
        "'{}' NOT LIKE '{}'".format(value_s, value_s + random_string(1, spaces=False)),
    ]
    return random.choice(tautologies)


# Return a random false string comparison, e.g. 'x'='y'.
def string_contradiction():
    value_s = random_string(random.randint(1, 5))
    contradictions = [
        "'{}'='{}'".format(value_s, value_s + random_string(1, spaces=False)),
        "'{}' LIKE '{}'".format(value_s, value_s + random_string(1, spaces=False)),
        "'{}'='{}'".format(value_s, value_s + random_string(1, spaces=False)),
        "'{}' LIKE '{}'".format(value_s, value_s + random_string(1, spaces=False)),
        "'{}'!='{}'".format(value_s, value_s),
        "'{}'<>'{}'".format(value_s, value_s),
        "'{}' NOT LIKE '{}'".format(value_s, value_s),
        "'{}'!='{}'".format(value_s, value_s),
        "'{}'<>'{}'".format(value_s, value_s),
        "'{}' NOT LIKE '{}'".format(value_s, value_s),
    ]
    return random.choice(contradictions)


# Return a random true numeric comparison, e.g. 5=5.
def num_tautology():
    value_n = random.randint(1, 10000)
    tautologies = [
        "{}={}".format(value_n, value_n),
        "{} LIKE {}".format(value_n, value_n),
        "{}!={}".format(value_n, value_n + 1),
        "{}<>{}".format(value_n, value_n + 1),
        "{} NOT LIKE {}".format(value_n, value_n + 1),
        "{} IN ({},{},{})".format(value_n, value_n - 1, value_n, value_n + 1),
    ]
    return random.choice(tautologies)


# Return a random false numeric comparison, e.g. 5=6.
def num_contradiction():
    value_n = random.randint(1, 10000)
    contradictions = [
        "{}={}".format(value_n, value_n + 1),
        "{} LIKE {}".format(value_n, value_n + 1),
        "{}!={}".format(value_n, value_n),
        "{}<>{}".format(value_n, value_n),
        "{} NOT LIKE {}".format(value_n, value_n),
        "{} NOT IN ({},{},{})".format(value_n, value_n - 1, value_n, value_n + 1),
    ]
    return random.choice(contradictions)