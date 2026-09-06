"""Fixed output-set contrast shared by authored and natural explanation probes."""
import numpy as np


def masses(probabilities, clause_ids, item_ids):
    numerator = float(probabilities[clause_ids].sum())
    denominator = 1.0 - numerator if item_ids is None else float(probabilities[item_ids].sum())
    if not 0 < numerator < 1 or not 0 < denominator <= 1:
        raise ValueError('Invalid output-set probability mass')
    return numerator, denominator


def contrast(probabilities, clause_ids, item_ids):
    numerator, denominator = masses(probabilities, clause_ids, item_ids)
    return float(np.log(numerator / denominator))
