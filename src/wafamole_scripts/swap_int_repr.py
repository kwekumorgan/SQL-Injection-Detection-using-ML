# swap_int_repr.py
# Vendored from WAF-A-MoLE (Demetrio et al., 2020), wafamole/payloadfuzzer/sqlfuzzer.py

import re
import random


# Replace one random numeric literal with an equivalent hex or subquery form.
def swap_int_repr(payload):
    candidates = list(re.finditer(r'\b\d+\b', payload))
    if not candidates:
        return payload

    candidate_pos = random.choice(candidates).span()
    candidate = payload[candidate_pos[0]:candidate_pos[1]]

    replacements = [
        hex(int(candidate)),
        "(SELECT {})".format(candidate),
    ]
    replacement = random.choice(replacements)

    return payload[:candidate_pos[0]] + replacement + payload[candidate_pos[1]:]