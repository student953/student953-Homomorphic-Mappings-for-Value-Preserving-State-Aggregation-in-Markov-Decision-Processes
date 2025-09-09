# Function used for general MDP calculus

import numpy as np
from utils.generic_model import GenericModel
from typing import Tuple, List


def norminf(value: np.ndarray) -> float:
    """
    Infinite norm of a vector.
    """
    return np.absolute(value).max()


def optimal_bellman_operator(
        env: GenericModel, value: np.ndarray, discount: float
) -> np.ndarray:
    """
    Returns T*(value)
    """
    Q = np.empty((env.action_dim, env.state_dim))
    for aa in range(env.action_dim):
        Q[aa] = env.reward_matrix[:, aa] + discount * env.transition_matrix[aa].dot(
            value
        )
    return Q.max(axis=0)


def q_optimal_bellman_operator(
        env: GenericModel, q_value: np.ndarray, discount: float
) -> np.ndarray:
    """
    Returns T*(q_value)
    """
    value = q_value.max(axis=1)
    q_value_new = np.zeros((env.state_dim, env.action_dim))
    for aa in range(env.action_dim):
        q_value_new[:, aa] = (
                env.reward_matrix[:, aa] + discount * env.transition_matrix[aa] @ value
        )
    return q_value_new


def bellman_operator(env: GenericModel, value: np.ndarray, discount: float):
    """
    Returns R + discount * T @ V
    """
    q_value = np.zeros((env.state_dim, env.action_dim))

    for aa in range(env.action_dim):
        q_value[:, aa] = (
                env.reward_matrix[:, aa]
                + discount * env.transition_matrix[aa] @ value
        )

    return q_value


def bellman_policy_operator(
        value: np.ndarray,
        discount: float,
        transition_policy: np.ndarray,
        reward_policy: np.ndarray,
) -> np.ndarray:
    """
    Returns R^pi + gamma . T^pi @ value
    """
    return reward_policy + discount * transition_policy.dot(value)


def iterative_policy_evaluation(
        transition_policy: np.ndarray,
        reward_policy: np.ndarray,
        discount: float,
        variation_tol: float,
        max_iteration_evaluation: int,
        initial_value: float = None,
) -> np.ndarray:
    """
    Apply Bellman^pi to initial_value until |initial_value - Bellman^pi initial_value| < tolerance
    """
    if initial_value is None:
        value = np.zeros((transition_policy.shape[0]))
    else:
        value = initial_value

    for i in range(max_iteration_evaluation):
        next_value = bellman_policy_operator(
            value, discount, transition_policy, reward_policy
        )
        distance = norminf(next_value - value)
        value = next_value

        if distance < variation_tol:
            break

    return value


def compute_transition_reward_policy(
        env: GenericModel, policy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given T, R, policy, returns T^pi, R^pi.
    """
    policy_t = policy.T[:, :, np.newaxis]
    matrix_all = []
    for a in env.transition_matrix:
        matrix_all.append(a.A[np.newaxis])
    matrix_all = np.vstack(matrix_all)
    transition_policy = (policy_t * matrix_all).sum(0)
    reward_policy = (policy * env.reward_matrix).sum(1)
    return transition_policy, reward_policy


def value_span_on_regions(
        full_value: np.ndarray, states_in_regions: List[List[int]]
) -> List[float]:
    """
    Returns [max full_value_k - min full_value_k for k in range(K)]
    """
    return [np.ptp(full_value[region]) for region in states_in_regions]


def q_value_span_on_regions(
        full_q_value: np.ndarray, states_in_regions: List[List[int]]
) -> np.ndarray:
    """
    Returns an array of shape (region_number, action_dim)
    """
    region_number, action_dim = len(states_in_regions), full_q_value.shape[1]
    spans_of_q = np.zeros((region_number, action_dim))
    for region_index, region in enumerate(states_in_regions):
        spans_of_q[region_index, :] = full_q_value[region].max(axis=0) - full_q_value[
            region
        ].min(axis=0)
    return spans_of_q


def get_full_phi_from_states_in_regions(
        env: GenericModel, states_in_region: List[List[int]]
):
    region_lengths = np.array([len(region) for region in states_in_region])

    # Initialize full_phi with zeros
    full_phi = np.zeros((env.state_dim, len(states_in_region)))

    # Use np.add.at for direct indexing and updating
    np.add.at(
        full_phi,
        (
            np.concatenate(states_in_region),
            np.repeat(np.arange(len(states_in_region)), region_lengths),
        ),
        1,
    )
    return full_phi


def get_weight_matrix_from_states_in_regions(
        state_dim: int, states_in_regions: List[List[int]], region_number: int
):
    """
    Get the w matrix s.t. w @ phi = 1 from states_in_regions.
    """
    weight_matrix = np.zeros((region_number, state_dim))
    for region_index, region in enumerate(states_in_regions):
        weight_matrix[region_index, region] = 1 / len(region)
    return weight_matrix


def get_weights_from_partial_phi(partial_phi: np.ndarray, check_inputs: bool = True):
    """
    Get the w matrix s.t. w @ phi = 1 from the partial_phi matrix.
    """
    if check_inputs:
        assert np.all(
            partial_phi.sum(axis=0) > 0
        ), "Each region should contain at least one state."
    weights = partial_phi.T
    weights = weights / weights.sum(axis=1)
    return weights


def get_value_policy_value(
        env: GenericModel, discount: float, value: np.ndarray, precision: float = 1e-3
) -> np.ndarray:
    """
    For any V, returns V pi V
    """
    policy = bellman_operator(env, value, discount).argmax(axis=1)
    transition_policy, reward_policy = compute_transition_reward_policy(env, policy)
    value_policy = iterative_policy_evaluation(
        transition_policy, reward_policy, discount, precision, value
    )
    return value_policy


def apply_obo_until_var_small(
        env: GenericModel,
        discount: float,
        variation_tol: float,
        initial_value: np.ndarray,
) -> Tuple[np.ndarray, float]:
    value = initial_value

    while True:
        new_value = optimal_bellman_operator(env, value, discount)
        variation = norminf(new_value - value)
        value = new_value
        if variation < variation_tol:
            break

    return value, variation


def inv_approximate(matrix: np.ndarray, order=5000):
    x = np.eye(matrix.shape[0]) - matrix
    val = x ** 0
    for i in range(order):
        val += x ** i
    return val
