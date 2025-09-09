from utils.generic_model import GenericModel
import numpy as np
from scipy.sparse import lil_matrix, lil_matrix
import time
from utils.calculus import (
    get_weight_matrix_from_states_in_regions,
    get_full_phi_from_states_in_regions,
)
from typing import List


class ValuePartition:
    def __init__(self, env: GenericModel, discount: float):
        self.env = env
        self.discount = discount

        self._initialize_trivial_aggregation()

    def divide_regions_along_tv(
        self,
        tv_vector: np.ndarray,
        epsilon: float,
        contracted_value_to_update: np.ndarray,
    ):
        new_partition_dictionary = self._build_new_partition_dictionary(
            tv_vector, epsilon
        )
        new_contracted_value = self._update_partition_and_value(
            new_partition_dictionary, contracted_value_to_update
        )
        self.aggregate_transition_matrix = None
        self.aggregate_reward_matrix = None
        self.full_phi = None
        self.weights = None

        return new_contracted_value

    def _build_new_partition_dictionary(self, tv_vector: np.ndarray, epsilon: float):
        new_partition_dictionary = {
            region_index: [region]
            for region_index, region in enumerate(self.states_in_region)
        }

        for region_index, region in enumerate(self.states_in_region):
            local_value = tv_vector[region]
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

    def _update_partition_and_value(
        self, new_partition_dictionary: dict, contracted_value_to_update: np.ndarray
    ) -> np.ndarray:
        """
        Update self.states_in_regions and return an extended version of contracted_value_to_update

        Inputs :
        new_partition_dictionary : {region_index : [new_region_1, new_region_2, ...]}
        contracted_value_to_update : (region_number, 1)
        Outputs :
        contracted_value_to_update : (new_region_number, 1)
        """
        for region_index, partition in new_partition_dictionary.items():
            self.states_in_region[region_index] = partition[0]
            self.states_in_region.extend(partition[1:])

            value_to_add = np.tile(
                contracted_value_to_update[region_index], (len(partition) - 1)
            )

            contracted_value_to_update = np.concatenate(
                (contracted_value_to_update, value_to_add), axis=0
            )

        return contracted_value_to_update

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

    def _initialize_trivial_aggregation(self):
        self.states_in_region = [list(range(self.env.state_dim))]

        self.aggregate_transition_matrix: np.ndarray = None
        self.aggregate_reward_matrix: np.ndarray = None
        self.full_phi: np.ndarray = None
        self.weights: np.ndarray = None
        self.current_contracted_value: np.ndarray = None

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
                self.env.transition_matrix[aa].dot(partial_phi)
                for aa in range(self.env.action_dim)
            ]

            self.aggregate_reward_matrix = self.env.reward_matrix

            return self.aggregate_transition_matrix, self.aggregate_reward_matrix

        elif (
            self.aggregate_transition_matrix is not None
            and self.aggregate_reward_matrix is not None
        ):
            # Already built, return it.
            return self.aggregate_transition_matrix, self.aggregate_reward_matrix
        else:
            assert False, "This case should not be seen."
