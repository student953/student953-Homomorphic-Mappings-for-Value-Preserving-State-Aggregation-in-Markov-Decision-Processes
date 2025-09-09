# Function used for partition calculus.

import numpy as np
from typing import List


def span_by_region(
    value_function: np.ndarray, states_in_region: List[List[int]]
) -> List[float]:
    """
    Return a list :
    [max(value[region])-min(value[region]) for region in regions]
    """
    return [np.ptp(value_function[region]) for region in states_in_region]
