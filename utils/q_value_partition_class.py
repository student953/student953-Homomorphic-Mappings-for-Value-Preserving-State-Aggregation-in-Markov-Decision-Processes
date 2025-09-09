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
from utils.calculus_projected import projected_optimal_q_bellman_operator
from typing import Tuple, List


class QValuePartition:
    """
    We assume a partition is described by
    """

    def __init__(self, env: GenericModel, discount) -> None:
        # Parsing arguments
        self.env = env
        self.discount = discount

        # Creating useful variables
        self.aggregate_transition_matrix = None
        self.aggregate_reward_matrix = None
        self.full_phi = None
        self.weights = None

        self.states_in_region = [list(range(self.env.state_dim))]
        self.full_phi = lil_matrix((self.env.state_dim, self.env.state_dim))
        self.full_phi[:, 0] = 1.0

    def projected_q_bellman_operator(self, contracted_q_value):
        """
        Get (Pi T_Q)(contracted_q_value)
        """
        self._compute_aggregate_transition_reward()

        return projected_optimal_q_bellman_operator(
            self.env,
            self.discount,
            contracted_q_value,
            self.aggregate_transition_matrix,
            self.aggregate_reward_matrix,
        )

    def _compute_max_span_indices(self, tq_vector: np.ndarray) -> Tuple:
        spans = np.max(tq_vector[self.states_in_region, :, None], axis=1) - np.min(
            tq_vector[self.states_in_region, :, None], axis=1
        )
        max_span = np.max(spans)
        region_index, action = np.unravel_index(np.argmax(spans), spans.shape)
        return max_span, region_index, action

    def divide_region_along_tq(
        self,
        tq_vector: np.ndarray,
        epsilon: float,
        contracted_q_value_to_update: np.ndarray,
    ):
        new_partition_dictionary = self._build_new_partition_dictionary(
            tq_vector, epsilon
        )
        new_contracted_q_value = self._update_partition_and_q_value(
            new_partition_dictionary, contracted_q_value_to_update
        )
        self.aggregate_transition_matrix = None
        self.aggregate_reward_matrix = None
        self.full_phi = None
        self.weights = None

        return new_contracted_q_value

    def _build_new_partition_dictionary(self, tq_vector: np.ndarray, epsilon: float):
        new_partition_dictionary = {
            region_index: [region]
            for region_index, region in enumerate(self.states_in_region)
        }
        for aa in range(self.env.action_dim):
            for region_index, region in enumerate(self.states_in_region):
                local_value = tq_vector[region, aa]
                if np.ptp(local_value) > epsilon:
                    new_partition = self._partition_along_value(
                        local_value, region, epsilon
                    )
                    if len(new_partition_dictionary[region_index]):
                        new_partition_dictionary[
                            region_index
                        ] = self._intersection_partitions(
                            new_partition_dictionary[region_index], list(new_partition)
                        )
                    else:
                        new_partition_dictionary[region_index] = new_partition
        return new_partition_dictionary

    def _intersection_partitions(
        self, partition_1: List[list], partition_2: List[list]
    ) -> list:
        return [
            list(set(sublist_1) & set(sublist_2))
            for sublist_1 in partition_1
            for sublist_2 in partition_2
            if set(sublist_1) & set(sublist_2)
        ]

    def _partition_along_value(
        self, local_value: np.ndarray, region: List[int], epsilon: float
    ):
        min_local_value = local_value.min()
        partition = {}
        for index, val in enumerate(local_value):
            region_state = region[index]
            interval_index = int((val - min_local_value) / epsilon)

            if interval_index in partition:
                partition[interval_index].append(region_state)
            else:
                partition[interval_index] = [region_state]
        return partition.values()

    def _update_partition_and_q_value(
        self, new_partition_dictionary: dict, contracted_q_value_to_update: np.ndarray
    ):
        for region_index, partition in new_partition_dictionary.items():
            self.states_in_region[region_index] = partition[0]
            self.states_in_region.extend(partition[1:])

            q_value_to_add = np.tile(
                contracted_q_value_to_update[region_index], (len(partition) - 1, 1)
            )

            contracted_q_value_to_update = np.concatenate(
                (contracted_q_value_to_update, q_value_to_add), axis=0
            )

        return contracted_q_value_to_update

    def _partial_phi(self) -> np.ndarray:
        if self.full_phi is None:
            self._build_full_phi()
        return self.full_phi[:, : len(self.states_in_region)]

    def _build_full_phi(self):
        if self.full_phi is None:
            self.full_phi = get_full_phi_from_states_in_regions(
                self.env, self.states_in_region
            )

    def _build_weights(self):
        if self.weights is None:
            self.weights = get_weight_matrix_from_states_in_regions(
                self.env.state_dim, self.states_in_region, len(self.states_in_region)
            )

    def _compute_aggregate_transition_reward(
        self,
    ) -> (list[lil_matrix], lil_matrix):
        """
        Get the aggregate transition function w @ T @ phi and the aggregate reward function w @ R.
        """
        if (
            self.aggregate_transition_matrix is None
            and self.aggregate_reward_matrix is None
        ):
            # We build it if None.
            partial_phi = self._partial_phi()
            self._build_weights()

            self.aggregate_transition_matrix = [
                self.weights @ self.env.transition_matrix[aa] @ partial_phi
                for aa in range(self.env.action_dim)
            ]

            self.aggregate_reward_matrix = self.weights.dot(self.env.reward_matrix)

            return self.aggregate_transition_matrix, self.aggregate_reward_matrix

        elif (
            self.aggregate_transition_matrix is not None
            and self.aggregate_reward_matrix is not None
        ):
            # Already built, return it.
            return self.aggregate_transition_matrix, self.aggregate_reward_matrix
        else:
            assert False, "This case should not be seen."
