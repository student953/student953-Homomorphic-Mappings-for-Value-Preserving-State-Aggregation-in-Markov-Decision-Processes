from utils.generic_model import GenericModel
import numpy as np
from typing import Optional


class GenericSolver:
    def __init__(self, env: GenericModel) -> None:
        self.runtime: Optional[float]
        self.name: str
        self.env = env
        self.value: np.ndarray

    def run(self):
        # Example
        pass

    def build_solution(self):
        # Example
        pass
