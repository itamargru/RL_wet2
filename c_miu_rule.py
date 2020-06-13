import numpy as np

action_number = 5
states_number = 2**action_number
jobs = np.arange(action_number)
miu = np.array([0.6, 0.5, 0.3, 0.7, 0.1])
cost = - np.array([[1, 4, 6, 2, 9]])

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

def calc_V_given_policy_numerical(policy, states, f, state_cost=None,  P=None, iteration=1000):
    P = P if P is not None else create_P_matrix_given_policy(policy, f)
    state_cost = state_cost if state_cost is not None else calculate_state_cost(policy, states)
    s = np.arange(states_number)

    V = np.zeros((states_number, 1))
    for i in np.arange(iteration):
        # V = states @ (cost / miu).T + P @ V
        V = state_cost + V[f[s, policy[s]]]
    return V


def create_P_matrix_given_policy(policy, f):
    P = np.zeros((states_number, states_number))
    for state in np.arange(states_number):
        P[state, f[state, policy[state]]] += 1
    return P


def calc_V_given_policy(policy, states, f, state_cost=None, P=None):
    P = P if P is not None else create_P_matrix_given_policy(policy, f)
    state_cost = state_cost if state_cost is not None else calculate_state_cost(policy, states)
    pinv_prob = np.linalg.pinv(np.eye(states_number)*(1 - 1e-5) - P)
    V = pinv_prob @ state_cost
    return V


def calculate_state_cost(policy, states):
    s = np.arange(states_number)
    miu_policy = miu[policy[s]]
    state_cost = (states @ cost.T).T / miu_policy
    return state_cost.T


def improve_policy_given_V(states, f, V, state_cost=None):
    m = miu * states
    m[m < 1e-3] = 1e-3
    state_cost = (states @ cost.T) / m if state_cost is None else state_cost
    V_a = state_cost + V[f].squeeze(axis=2)
    policy = np.argmax(V_a, axis=1)
    return policy


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

    return np.array(policy, dtype=int)


def get_optimal_policy(states):
    cost_state = np.repeat(cost, states_number, axis=0)
    cost_state = (cost_state * miu) * states
    policy = np.argmin(cost_state, axis=1)
    return policy

def max_cost_policy():
    policy = []
    for state in states:
        max_val = np.argmax(state*cost)
    policy += [max_val]
    return policy
def policy_iteration(init_policy, states,  f, state_cost=None, P=None):
    V_prev = np.zeros((states_number, 1))
    V_curr = np.ones((states_number, 1))
    policy = init_policy
    while (V_prev != V_curr).any():
        V_prev = V_curr
        V_curr = calc_V_given_policy_numerical(policy, states, f, state_cost=state_cost, P=P)
        policy = improve_policy_given_V(states, f, V, state_cost=state_cost)
    return policy

# policy_0 = [0, 0, 0, 1, 0, 4, 2, 3, 0, 8, 2, 9, 8, 9, 10, 14, 0, 16, 2, 17, 16, 17, 6, 7, 8, 24, 10, 25, 20, 21, 28, 27]

if __name__ == "__main__":
    states = initial_states()
    f = matrix_next_state_given_state_policy()
    policy_opt = get_optimal_policy(states)
    policy_init = np.array([0, 0, 1, 1, 2, 2, 1, 1, 3, 3, 1, 1, 2, 2, 1, 1, 4, 4, 1, 1, 2, 2, 4, 4, 3, 3, 4, 0, 4, 4, 2, 2], dtype=int)
    V = calc_V_given_policy(policy_init, states, f)
    policy = policy_iteration(policy_init, states, f, V)
    # V2 = calc_V_given_policy_numerical(policy, states, f)
    pass
