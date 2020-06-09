import numpy as np
import matplotlib.pyplot as plt

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

def calc_V_given_policy(policy, states, iteration=100):
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



action_number = 5
states_number = 2**action_number
jobs = np.arange(action_number)
miu = np.array([0.6, 0.5, 0.3, 0.7, 0.1])
cost = np.array([[1, 4, 6, 2, 9]])
states = initial_states()


def clear_bit(num, bit):
    return num & (~(1 << bit))

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

def plot_policy(policy):

    # policy = policy[player_edge[0] : player_edge[1] + 1, dealer_edge[0]:]
    fig, ax = plt.subplots()
    ax.imshow(policy, origin='lower')
    # ax.set_xticks(np.arange(dealer_edge[1] - dealer_edge[0] + 1))
    # ax.set_xticklabels(np.arange(dealer_edge[0], dealer_edge[1] + 1))
    # ax.set_yticks(np.arange(player_edge[1] - player_edge[0] + 1))
    # ax.set_yticklabels(np.arange(player_edge[0], player_edge[1] + 1))

    ax.set_xlabel("dealer's card value")
    ax.set_ylabel("player's sum of cards value")

    txt_hit = 'Hit Zone'
    txt_stick = 'Stick Zone'
    # plt.text(3, 15, txt_stick, color="white")
    # plt.text(3, 1, txt_hit)

    plt.savefig("Policy_Plot_server.png")
    plt.show()

if __name__ == "__main__":
    # policy = [0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1,
    #            0]
    policy = max_cost_policy()
    V = calc_V_given_policy(policy, states)
    plot_policy(policy)
    pass
