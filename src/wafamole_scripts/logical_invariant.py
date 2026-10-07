# logical_invariant.py
# Vendored from WAF-A-MoLE (Demetrio et al., 2020), wafamole/payloadfuzzer/sqlfuzzer.py

import re
import random
from .fuzz_utils import num_tautology, string_tautology, num_contradiction, string_contradiction


# Append a redundant true/false clause after an existing tautology or contradiction.
def logical_invariant(payload):
    num_tautologies_pos = list(re.finditer(r'\b(\d+)(\s*=\s*|\s+(?i:like)\s+)\1\b', payload))
    num_tautologies_neg = list(re.finditer(r'\b(\d+)(\s*(!=|<>)\s*|\s+(?i:not like)\s+)(?!\1\b)\d+\b', payload))
    string_tautologies_pos = list(re.finditer(r'(\'|\")([a-zA-Z]{1}[\w#@$]*)\1(\s*=\s*|\s+(?i:like)\s+)(\'|\")\2\4', payload))
    string_tautologies_neg = list(re.finditer(r'(\'|\")([a-zA-Z]{1}[\w#@$]*)\1(\s*(!=|<>)\s*|\s+(?i:not like)\s+)(\'|\")(?!\2)([a-zA-Z]{1}[\w#@$]*)\5', payload))

    # Only fires if a real tautology or contradiction already exists.
    results = num_tautologies_pos + num_tautologies_neg + string_tautologies_pos + string_tautologies_neg
    if not results:
        return payload

    candidate = random.choice(results)
    pos = candidate.end()

    replacement = random.choice([
        " AND 1", " AND True", " AND " + num_tautology(), " AND " + string_tautology(),
        " OR 0", " OR False", " OR " + num_contradiction(), " OR " + string_contradiction(),
    ])

    return payload[:pos] + replacement + payload[pos:]