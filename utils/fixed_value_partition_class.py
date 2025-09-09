from utils.generic_model import GenericModel
import numpy as np
from scipy.sparse import lil_matrix, lil_matrix
import time
from utils.calculus import (
    norminf,
    get_weights_from_partial_phi,
    get_weight_matrix_from_states_in_regions,
    get_full_phi_from_states_in_regions,
)

from utils.calculus_projected import projected_optimal_bellman_operator


def afficher(texte, valeur, arrondir=False):
    print()
    print(texte)
    if arrondir:
        valeur = np.round(valeur, 2)
    print(valeur)
    print()


class FixedValuePartition:
    def __init__(
        self, env: GenericModel, discount: float, fixed_number_of_regions: int = 100
    ) -> None:
        self.env = env
        self.discount = discount
        self.fixed_number_of_regions = fixed_number_of_regions

        self._initialize_trivial_aggregation()

    def projected_bellman_operator(self, contracted_value):
        """
        Get (Pi T)(contracted_value)
        """
        if self.aggregate_reward_matrix is None:
            self._build_aggregate_transition_reward()

        self._build_weights()

        return projected_optimal_bellman_operator(
            self.env,
            self.discount,
            contracted_value,
            self.aggregate_transition_matrix,
            self.aggregate_reward_matrix,
            self.weights,
        )

    def build_partition_along_value(
        self,
        local_value: np.ndarray,
    ):
        self._initialize_trivial_aggregation()
        region = list(range(self.env.state_dim))

        max_local_value = local_value.max()
        min_local_value = local_value.min()

        epsilon = (max_local_value - min_local_value) / self.fixed_number_of_regions
        intervals = [
            min_local_value + step * epsilon
            for step in range(self.fixed_number_of_regions)
        ]
        indices = np.searchsorted(intervals, local_value)

        new_region_dictionary = {region_name: [] for region_name in set(indices)}
        for state_index_in_region_list, state in enumerate(region):
            region_name = indices[state_index_in_region_list]
            new_region_dictionary[region_name].append(state)

        self._divide_region_multiple_pieces(list(new_region_dictionary.values()))

    def _divide_region_multiple_pieces(
        self,
        new_partition_of_region: list[list[int]],
    ):
        """
        Update of self.states_in_region.
        """
        self.region_number = len(new_partition_of_region)
        self.states_in_region = new_partition_of_region

    def _partial_phi(self) -> np.ndarray:
        if self.full_phi is None:
            self._build_full_phi()
        return self.full_phi[:, : self.region_number]

    def _build_full_phi(self):
        if self.full_phi is None:
            self.full_phi = get_full_phi_from_states_in_regions(
                self.env, self.states_in_region
            )

    def _build_weights(self):
        if self.weights is None:
            self.weights = get_weight_matrix_from_states_in_regions(
                self.env.state_dim, self.states_in_region, self.region_number
            )

    def _initialize_trivial_aggregation(self):
        self.region_number = 1
        self.states_in_region = [list(range(self.env.state_dim))]

        self.aggregate_transition_matrix: np.ndarray = None
        self.aggregate_reward_matrix: np.ndarray = None
        self.full_phi: np.ndarray = None
        self.weights: np.ndarray = None
        self.current_contracted_value: np.ndarray = None

    def _build_aggregate_transition_reward(
        self,
    ) -> (list[lil_matrix], lil_matrix):
        """
        Get the aggregate transition function w @ T @ phi and the aggregate reward function w @ R.
        """
        if (
            self.aggregate_transition_matrix is None
            and self.aggregate_reward_matrix is None
        ):
            self._build_full_phi()
            partial_phi = self._partial_phi()

            self.aggregate_transition_matrix = [
                self.env.transition_matrix[aa] @ partial_phi
                for aa in range(self.env.action_dim)
            ]
            self.aggregate_reward_matrix = self.env.reward_matrix

        elif (
            self.aggregate_transition_matrix is not None
            and self.aggregate_reward_matrix is not None
        ):
            return

        else:
            assert False, "One of the two matrices is None and not the other one."
