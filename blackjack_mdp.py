from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

X_states = range(4, 31 + 1)
Y_states = range(2, 11 + 1)
dealer_states = range(2, 27 + 1)


cards = range(2, 14 + 1)
cards_value = list(range(2, 10 + 1)) + [10]*3 + [11]
cards_value = dict(zip(cards, cards_value))

prob_value = np.ones((10, 1))
prob_value[-2] = 4
prob_value = prob_value / len(cards)


def fill_result_prob_matrix(func):
    prob = np.zeros(( X_states[-1] + 1, dealer_states[-1] + 1 ))
    for x in X_states:
        for k in reversed(dealer_states):
            prob[x, k] = func(prob, x, k)
    return prob

def win_func(prob_matrix, x, k):
    if k > 21 or (x > k and 17 <= k <= 21 and x <= 21):
        return 1
    elif x > 21 or (x <= k and 17 <= k <= 21):
        return 0
    else:
        return prob_matrix[x, (k + cards_value[cards[0]]): (k + cards_value[cards[-1]] + 1)].dot(prob_value)

def lose_func(prob_matrix, x, k):
    if k > 21 or (x >= k and 17 <= k <= 21 and x <= 21):
        return 0
    elif (x > 21) or (x < k and 17 <= k <= 21):
        return 1
    else:
        return prob_matrix[x, (k + cards_value[cards[0]]): (k + cards_value[cards[-1]] + 1)].dot(prob_value)

def fill_reward():
    win_matrix = fill_result_prob_matrix(win_func)
    lose_matrix = fill_result_prob_matrix(lose_func)
    draw_matrix = 1 - win_matrix - lose_matrix
    reward = win_matrix - lose_matrix
    return reward

def solve_blackjack(iterations=1000):
    reward = fill_reward()
    V = np.zeros((X_states[-1] + 1, cards_value[cards[-1]] + 1))
    policy = V
    for i in range(iterations):
        for y in Y_states:
            for x in reversed(X_states):
                if x >= 20:
                    V[x, y] = reward[x, y]
                    policy[x, y] = 0
                else:
                    c = V[(x + cards_value[cards[0]]): (x + cards_value[cards[-1]] + 1), y].dot(prob_value)[0]
                    actions_value = np.array([reward[x, y], c])
                    V[x, y] = np.max(actions_value)
                    policy[x, y] = np.argmax(actions_value)
    return V, policy

def plot_policy(policy):
    player_edge = [4, 21]
    dealer_edge = [2, 11]
    policy = policy[player_edge[0] : player_edge[1] + 1, dealer_edge[0]:]
    fig, ax = plt.subplots()
    ax.imshow(policy, origin='lower')
    ax.set_xticks(np.arange(dealer_edge[1] - dealer_edge[0] + 1))
    ax.set_xticklabels(np.arange(dealer_edge[0], dealer_edge[1] + 1))
    ax.set_yticks(np.arange(player_edge[1] - player_edge[0] + 1))
    ax.set_yticklabels(np.arange(player_edge[0], player_edge[1] + 1))
    plt.show()


def plot_V_function(V):
    player_edge = [4, 21]
    dealer_edge = [2, 11]
    V = V[player_edge[0] : player_edge[1] + 1, dealer_edge[0]:]
    X, Y = np.meshgrid(np.arange(dealer_edge[0], dealer_edge[1] + 1), np.arange(player_edge[1] - player_edge[0] + 1))
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, V)
    plt.show()


def main():
    V, policy = solve_blackjack()
    plot_V_function(V)
    plot_policy(policy)
    a = 0


if __name__ == "__main__":
    main()

