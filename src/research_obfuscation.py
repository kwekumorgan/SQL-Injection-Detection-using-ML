# research_obfuscation.py
# Stacks 1-4 obfuscation techniques randomly per row, applied to MALICIOUS rows only.
# Benign rows pass through unchanged so the output keeps both classes.

import random
from collections import Counter
import pandas as pd
from . import config
from .sqlmap_scripts import randomcase, space2comment, randomcomments
from .wafamole_scripts.logical_invariant import logical_invariant
from .wafamole_scripts.swap_int_repr import swap_int_repr


# One of the two sqlmap comment techniques, picked at random.
def comment_inject(query, rng):
    tamper_func = rng.choice([space2comment.tamper, randomcomments.tamper])
    return tamper_func(query)


TECHNIQUES = {
    "case_swap": lambda q, rng: randomcase.tamper(q),
    "comment_inject": comment_inject,
    "integer_encode": lambda q, rng: swap_int_repr(q),
    "logical_invariant": lambda q, rng: logical_invariant(q),
}


# Stack 1-4 random techniques, in random order, on one query.
def stack_techniques(query, rng):
    n = rng.randint(1, 4)
    chosen = rng.sample(list(TECHNIQUES.keys()), n)

    result = query
    for name in chosen:
        result = TECHNIQUES[name](result, rng)

    return result, chosen


# Print summary stats: technique usage counts, stack-size distribution, sample rows.
def print_summary(result_df):
    print(f"\nTotal rows: {len(result_df):,}")
    print(f"Class breakdown: {dict(result_df['Label'].value_counts())}")

    print("\nStack size distribution (number of techniques applied per row):")
    for n, count in sorted(Counter(result_df["num_techniques"]).items()):
        print(f"  {n} technique(s): {count:,} rows")

    technique_counts = Counter()
    for combo in result_df["techniques_used"]:
        if not combo:          # skip untouched benign rows
            continue
        technique_counts.update(combo.split("+"))

    print("\nHow often each technique was used (across all rows, any position in the stack):")
    for name, count in technique_counts.most_common():
        print(f"  {name}: {count:,}")

    print("\nSample rows:")
    for _, row in result_df.sample(n=min(5, len(result_df)), random_state=1).iterrows():
        print(f"\n  Label: {row['Label']}  Techniques: {row['techniques_used']}")
        print(f"  Result: {row['Query']}")


# Obfuscate malicious rows; leave benign rows untouched.
def build_research_obfuscated_set(random_state=None):
    random_state = config.RANDOM_STATE if random_state is None else random_state
    rng = random.Random(random_state)

    df = pd.read_csv(config.TEST_CLEAN_PATH)
    print(f"Loaded {len(df):,} rows from test_clean.csv")

    n_malicious = (df["Label"] == 1).sum()
    n_benign = (df["Label"] == 0).sum()
    print(f"  Malicious (to be obfuscated): {n_malicious:,}")
    print(f"  Benign (left untouched):      {n_benign:,}")

    records = []
    for _, row in df.iterrows():
        original_query = str(row["Query"])

        if row["Label"] == 1:
            obfuscated_query, techniques_used = stack_techniques(original_query, rng)
        else:
            obfuscated_query = original_query
            techniques_used = []

        records.append({
            "Query": obfuscated_query,
            "Label": row["Label"],
            "techniques_used": "+".join(techniques_used),
            "num_techniques": len(techniques_used),
        })

    result_df = pd.DataFrame.from_records(records)

    print_summary(result_df)

    result_df.to_csv(config.RESEARCH_OBFUSCATED_PATH, index=False)
    print(f"\nSaved -> {config.RESEARCH_OBFUSCATED_PATH}")

    return result_df


if __name__ == "__main__":
    build_research_obfuscated_set()