# encoding: utf-8

import numpy as np


def value_function_exact(P, R, gamma):
    """@todo: Docstring for value_function.
    :returns: @todo

    """
    I = np.eye(len(R))
    return np.linalg.inv(I - gamma * P.T) @ R


def _value_function_iterator(P, R, gamma):
    """Estimate value function by iterating the expected Bellman equation
    synchronously

    :returns: @todo

    """
    value = R
    while True:
        yield value
        value = R + gamma * P.T @ value


def value_function(P, R, gamma, thresh=1e-6):
    iterator = _value_function_iterator(P, R, gamma)
    value = next(iterator)
    for new_value in iterator:
        if np.linalg.norm(value - new_value, ord=np.inf) < thresh:
            return new_value
        value = new_value


def mrp_from_mdp(P_mdp, R_mdp, policy):
    """@todo: Docstring for mrp_from_mdp.

    :param P_mdp: (actions, target_state, source_state)
    :param R_mdp: (actions, source_state)
    :param policy: (actions, source_state)
    :returns: @todo

    """
    P_mrp = np.sum(policy[:, None, :] * P_mdp, axis=0)
    R_mrp = np.sum(policy * R_mdp, axis=0)
    return P_mrp, R_mrp


def greedy_policy(P, R, v):
    """@todo: Docstring for greedy_policy.

    :param vfunc: @todo
    :returns: @todo

    """
    expected_returns = R + np.tensordot(P, v, axes=(1, 0))
    policy = np.zeros_like(expected_returns)
    for p_s, idx in zip(policy.T, np.argmax(expected_returns, axis=0)):
        p_s[idx] = 1
    return policy


def _optimal_policy_iterator(P, R, gamma, policy_init=None, v_thresh=1e-6):
    """@todo: Docstring for _policy_iterator.

    :param P: @todo
    :param R: @todo
    :param policy_init: @todo
    :returns: @todo

    """
    # initialize with fully random policy
    policy = policy_init if policy_init is not None else np.ones(R.shape) / len(R)
    while True:
        P_policy, R_policy = mrp_from_mdp(P, R, policy)
        v_policy = value_function(P_policy, R_policy, gamma, thresh=v_thresh)
        yield policy, v_policy
        policy = greedy_policy(P, R, v_policy)


def _optimal_value_iteration(P, R, gamma):
    """@todo: Docstring for _optimal_value_iteration.

    :param P: @todo
    :param R: @todo
    :param gamma: @todo
    :returns: @todo

    """
    v = np.zeros(R.shape[1])
    while True:
        expected_returns = R + np.tensordot(P, v, axes=(1, 0))
        v = np.max(expected_returns, axis=0)
        yield v
