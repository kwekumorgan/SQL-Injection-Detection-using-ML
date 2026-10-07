# number_randomization.py
# Randomizes numeric literals in query text to prevent Chi-square from
# finding a spurious statistical association with any specific number
# (originally discovered with "1" dominating malicious training rows).




import re
import random


COMBINED_NUMBER_PATTERN = re.compile(r"\b(\d+)\s*=\s*\1\b|\b\d+\b")

RANDOM_LOW = 1
RANDOM_HIGH = 999


# Identify spans of text that must NOT be randomized: select-list
# column enumeration and CHAR()/CHR() encoding calls.
def find_protected_spans(text):
    spans = []

    # Comma-separated numeric list, ONLY when it follows 'select'.
    for m in re.finditer(r"(?i:\bselect\s+)(\d+(?:\s*,\s*\d+)+)", text):
        spans.append((m.start(1), m.end(1)))

    # CHAR(...)/CHR(...) encoding calls, single OR multi-arg form.
    for m in re.finditer(r"(?i:\b(?:char|chr)\s*\(\s*\d+(?:\s*,\s*\d+)*\s*\))", text):
        spans.append((m.start(), m.end()))

    return spans


def is_protected(start, end, spans):
    return any(s <= start and end <= e for s, e in spans)


# Hands out a fresh random integer per call. One instance per class
# (malicious / benign), each independently seeded but sharing the
# same (low, high) range.
class RandomNumberGenerator:
    def __init__(self, seed, low=RANDOM_LOW, high=RANDOM_HIGH):
        self.rng = random.Random(seed)
        self.low = low
        self.high = high

    def next_number(self):
        return str(self.rng.randint(self.low, self.high))


# Replace every non-protected numeric literal in a single query with
# a random value, sharing one value across both sides of a tautology.
def randomize_numbers(query_text, generator):
    text = str(query_text)
    protected_spans = find_protected_spans(text)

    def _replace(match):
        if is_protected(match.start(), match.end(), protected_spans):
            return match.group(0)  # structural/encoding context -- untouched
        if match.group(1) is not None:
            # Tautology match -- ONE shared value on both sides.
            val = generator.next_number()
            return f"{val} = {val}"
        return generator.next_number()  # generic standalone number

    return COMBINED_NUMBER_PATTERN.sub(_replace, text)


# Apply randomization to an entire DataFrame, using two independent
# generators keyed by class label.
def apply_randomization(df, seed_malicious, seed_benign, text_col="Query", label_col="Label"):
    gen_malicious = RandomNumberGenerator(seed=seed_malicious)
    gen_benign = RandomNumberGenerator(seed=seed_benign)

    def _row(row):
        generator = gen_malicious if row[label_col] == 1 else gen_benign
        return randomize_numbers(row[text_col], generator)

    df = df.copy()
    df[text_col] = df.apply(_row, axis=1)
    return df