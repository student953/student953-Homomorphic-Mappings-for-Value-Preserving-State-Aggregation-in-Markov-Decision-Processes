# Function used for general MDP calculus with projections

import numpy as np
from utils.generic_model import GenericModel
from typing import List, Tuple
from utils.calculus import norminf


def projected_optimal_bellman_operator(
    env: GenericModel,
    discount: float,
    contracted_value: np.ndarray,
    aggregated_transition: List[np.ndarray],
    aggregated_reward: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Returns w @ max_a(agg_R + gamma * agg_T @ agg_V)
    """
    region_number = contracted_value.shape[0]
    assert len(aggregated_transition) == env.action_dim
    assert aggregated_transition[0].shape == (env.state_dim, region_number)
    assert aggregated_reward.shape == (env.state_dim, env.action_dim)

    contracted_q_value = np.empty((env.state_dim, env.action_dim))
    for aa in range(env.action_dim):
        contracted_q_value[:, aa] = aggregated_reward[
            :, aa
        ] + discount * aggregated_transition[aa].dot(contracted_value)
    contracted_value = weights.dot((contracted_q_value).max(axis=1))
    return contracted_value


def apply_pobo_until_var_small(
    env: GenericModel,
    discount: float,
    aggregated_transition: List[np.ndarray],
    aggregated_reward: np.ndarray,
    weights: np.ndarray,
    variation_tol: float,
    initial_contracted_value: np.ndarray = None,
):
    if initial_contracted_value is None:
        contracted_value = np.zeros((aggregated_reward.shape[0]))
    else:
        contracted_value = initial_contracted_value

    while True:
        new_contracted_value = projected_optimal_bellman_operator(
            env,
            discount,
            contracted_value,
            aggregated_transition,
            aggregated_reward,
            weights,
        )
        variation = norminf(new_contracted_value - contracted_value)
        contracted_value = new_contracted_value
        if variation < variation_tol:
            break

    return contracted_value, variation


def projected_optimal_q_bellman_operator(
    env: GenericModel,
    discount: float,
    contracted_q_value: np.ndarray,
    aggregated_transition: List[np.ndarray],
    aggregated_reward: np.ndarray,
) -> np.ndarray:
    """
    Returns w @ max_a(agg_R + gamma * agg_T @ agg_V)
    """
    region_number = contracted_q_value.shape[0]
    # assert len(aggregated_transition) == env.action_dim
    # assert aggregated_transition[0].shape == (region_number, region_number)
    # assert aggregated_reward.shape == (region_number, env.action_dim)

    contracted_value = contracted_q_value.max(axis=1)

    contracted_q_value = np.empty((region_number, env.action_dim))
    for aa in range(env.action_dim):
        contracted_q_value[:, aa] = aggregated_reward[
            :, aa
        ] + discount * aggregated_transition[aa].dot(contracted_value)

    return contracted_q_value


def apply_poqbo_until_var_small(
    env: GenericModel,
    discount: float,
    aggregated_transition: List[np.ndarray],
    aggregated_reward: np.ndarray,
    variation_tol: float,
    initial_q_contracted_value: np.ndarray = None,
) -> np.ndarray:
    if initial_q_contracted_value is None:
        q_contracted_value = np.zeros((aggregated_reward.shape[0]))
    else:
        q_contracted_value = initial_q_contracted_value

    while True:
        new_q_contracted_value = projected_optimal_q_bellman_operator(
            env,
            discount,
            q_contracted_value,
            aggregated_transition,
            aggregated_reward,
        )
        variation = norminf(new_q_contracted_value - q_contracted_value)
        q_contracted_value = new_q_contracted_value
        if variation < variation_tol:
            break

    return q_contracted_value, variation


def projected_policy_bellman_operator(
    discount: float,
    contracted_value: np.ndarray,
    aggregated_transition_policy: np.ndarray,
    aggregated_reward_policy: np.ndarray,
) -> np.ndarray:
    """
    Returns (w R^pi phi) + discount * (w T^pi phi) V
    """
    return aggregated_reward_policy + discount * aggregated_transition_policy.dot(
        contracted_value
    )


def apply_ppbo_until_var_small(
    discount: float,
    aggregated_transition_policy: List[np.ndarray],
    aggregated_reward_policy: np.ndarray,
    variation_tol: float,
    initial_contracted_value: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Applies Pi T^pi until residual smaller than variation_tol.
    """
    contracted_value = initial_contracted_value

    while True:
        new_contracted_value = projected_policy_bellman_operator(
            discount,
            contracted_value,
            aggregated_transition_policy,
            aggregated_reward_policy,
        )
        variation = norminf(new_contracted_value - contracted_value)
        contracted_value = new_contracted_value
        if variation < variation_tol:
            break

    return variation, contracted_value
