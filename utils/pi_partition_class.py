import numpy as np
from scipy.sparse import lil_matrix
import time
from typing import List
from utils.generic_model import GenericModel
from utils.calculus import get_full_phi_from_states_in_regions


def norminf(value):
    return np.linalg.norm(value, ord=np.inf)


def get_weight_matrix_from_states_in_regions(
    state_dim: int, states_in_regions: list[list[int]], region_number: int
):
    """
    Get the w matrix s.t. w @ phi = 1
    """
    K = region_number
    weight_matrix = lil_matrix((K, state_dim))
    for region_index, region in enumerate(states_in_regions):
        value = 1 / len(region)
        weight_matrix[region_index, region] = value
    return weight_matrix


def get_weights_from_partial_phi(partial_phi):
    """
    Get the w matrix s.t. w @ phi = 1
    """
    weights = partial_phi.T
    weights = weights / weights.sum(axis=1)[:, np.newaxis]
    return weights


class PiPartition:
    """
    We assume a partition is described by
    """

    def __init__(self, env: GenericModel, discount) -> None:
        self.env = env
        self.discount = discount

        self._initialize_trivial_aggregation()

    def projected_bellman_policy_operator(self, contracted_value: np.ndarray):
        """
        Get (Pi T)(contracted_value)
        """
        # self._compute_aggregate_transition_reward_policy()
        self._build_weights()

        contracted_value = (
            self.aggregate_reward_policy
            + self.discount * self.aggregate_transition_policy.dot(contracted_value)
        )
        return contracted_value

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
        self.full_phi = lil_matrix((self.env.state_dim, self.env.state_dim))
        self.states_in_region = [list(range(self.env.state_dim))]

        self.full_phi = None
        self.weights = None
        self.aggregate_transition_policy = None
        self.aggregate_reward_policy = None

    def update_transition_reward_policy(
        self, transition_policy: np.ndarray, reward_policy: np.ndarray
    ):
        self.transition_policy = transition_policy
        self.reward_policy = reward_policy
        self._compute_aggregate_transition_reward_policy()

    def _compute_aggregate_transition_reward_policy(
        self, recompute=False
    ) -> (np.ndarray, np.ndarray):
        """
        Returns (w @ T^pi @ phi), (w @ R^pi)
        """
        if (
            self.aggregate_transition_policy is None
            and self.aggregate_reward_policy is None
        ) or recompute:
            partial_phi = self._partial_phi()
            self._build_weights()

            self.aggregate_transition_policy = self.weights.dot(
                self.transition_policy.dot(partial_phi)
            )
            self.aggregate_reward_policy = self.weights.dot(self.reward_policy)

            return self.aggregate_transition_policy, self.aggregate_reward_policy

        elif (
            self.aggregate_transition_policy is not None
            and self.aggregate_reward_policy is not None
        ):
            return self.aggregate_transition_policy, self.aggregate_reward_policy

        else:
            assert False, "Impossible case."
