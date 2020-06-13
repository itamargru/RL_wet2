import numpy as np
import matplotlib.pyplot as plt

action_number = 5
states_number = 2**action_number
jobs = np.arange(action_number)
miu = np.array([0.6, 0.5, 0.3, 0.7, 0.1])
cost = np.array([[1, 4, 6, 2, 9]])

def initial_states():
    states = []
    for i in np.arange(2 ** action_number):
        arr = np.zeros(action_number)
        for j in np.arange(action_number):
            k = i >> j
            arr[j] = (k & 1)
        states += [arr]
    states = np.stack(states)
    return states

# V_p(s) = sum_s(cost) + miu(a) * V_p(s \ a) + (1 - miu(a)) * V_p(s)

def calc_V_given_policy_numerical(policy, states, iteration=100):
    P = create_P_matrix_given_policy(policy)
    V = np.linalg.pinv(np.eye(states_number) - P) @ (states @ cost.T)

    V_prev = np.zeros(states_number)
    V_curr = np.zeros(states_number)
    for i in np.arange(iteration):
        for state in np.arange(states_number):
            if state == 0:
                V_curr[state] = 0
            next_state = state & (~(1 << policy[state]))
            V_curr[state] = (states[state] @ cost.T)[0] + miu[policy[state]] * V_prev[next_state] + \
                (1 - miu[policy[state]]) * V_prev[state]
    return V_curr


def create_P_matrix_given_policy(policy, f):
    P = np.zeros((states_number, states_number))
    for state in np.arange(states_number):
        P[state, f[state, policy[state]]] += miu[policy[state]]
        P[state, state] += 1 - miu[policy[state]]
    return P


def calc_V_given_policy(policy, states, f):
    P = create_P_matrix_given_policy(policy, f)
    state_cost = states @ cost.T
    pinv_prob = np.linalg.pinv(np.eye(states_number) - P)
    V = pinv_prob @ state_cost
    return V


def clear_bit(num, bit):
    return num & (~(1 << bit))


def matrix_next_state_given_state_policy():
    f = np.zeros((states_number, action_number), dtype=int)
    for state in range(states_number):
        for action in range(action_number):
            f[state, action] = clear_bit(state, action)
    return f


def random_policy():
    policy = [0]
    for i in np.arange(1, 2**action_number):
        p = i
        while p == i:
            k = np.random.randint(0, action_number)
            p = clear_bit(i, k)
        policy += [k]
    return policy

def max_cost_policy():
    policy = []
    for state in states:
        max_val = np.argmax(state * cost)
        policy += [max_val]
    return policy



if __name__ == "__main__":
    # policy = [0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1,
    #            0]
    states = initial_states()
    f = matrix_next_state_given_state_policy()

    policy = max_cost_policy()
    V = calc_V_given_policy(policy, states)
    pass
