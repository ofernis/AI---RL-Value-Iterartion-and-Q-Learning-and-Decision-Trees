import copy
from copy import deepcopy
import random
# import termcolor
import numpy as np

def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    delta = 0
    U = np.zeros((mdp.num_row, mdp.num_col))
    U_prime = copy.deepcopy(U_init)
    while True:
        delta = 0
        U = copy.deepcopy(U_prime)
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                if mdp.board[r][c] == 'WALL':
                    U_prime[r][c] = 'WALL'
                elif (r, c) in mdp.terminal_states:
                    U_prime[r][c] = float(mdp.board[r][c])
                else:
                    actions_enum = {0: 'UP', 1: 'DOWN', 2: 'RIGHT', 3: 'LEFT'}
                    s = (r, c)  # current state s
                    action_expectations = []

                    for action, probs in mdp.transition_function.items():
                        util_expectation = 0
                        for prob_act_index, prob in enumerate(probs):
                            s_prime = mdp.step(s, actions_enum[prob_act_index])  # s_prime = (r',c')
                            util_expectation += prob * U[s_prime[0]][s_prime[1]]
                        action_expectations.append(util_expectation)

                    U_prime[r][c] = float(mdp.board[r][c]) + mdp.gamma * max(action_expectations)
                    delta = max(delta, abs(U_prime[r][c] - U[r][c]))
        if delta < (epsilon * ((1 - mdp.gamma) / mdp.gamma)) or (mdp.gamma == 1 and delta == 0):
            break
    return U
    # ========================


def get_policy(mdp, U):
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    policy = []
    actions = list(mdp.actions.keys())
    for r in range(mdp.num_row):
        cur_policy = []
        for c in range(mdp.num_col):
            if (r, c) in mdp.terminal_states:
                cur_policy.append(float(mdp.board[r][c]))
                continue
            elif mdp.board[r][c] == "WALL":
                cur_policy.append("WALL")
                continue
            value_max = float('-inf')
            max_act = None
            for a in actions:
                current_expectation = 0
                for i, sum_action in enumerate(actions):
                    prob = mdp.transition_function[a][i]
                    s_prime = mdp.step((r, c), sum_action)
                    current_expectation += prob * float(U[s_prime[0]][s_prime[1]])
                if current_expectation >= value_max:
                    value_max = current_expectation
                    max_act = a
            cur_policy.append(max_act)
        policy.append(cur_policy)
    return policy
    # ========================

def one_dim_conversion(state, col_num):
    return state[0] * col_num + state[1]

def q_learning(mdp, init_state, total_episodes=10000, max_steps=999, learning_rate=0.7, epsilon=1.0,
                      max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.8):
    # TODO:
    # Given the mdp and the Qlearning parameters:
    # total_episodes - number of episodes to run for the learning algorithm
    # max_steps - for each episode, the limit for the number of steps
    # learning_rate - the "learning rate" (alpha) for updating the table values
    # epsilon - the starting value of epsilon for the exploration-exploitation choosing of an action
    # max_epsilon - max value of the epsilon parameter
    # min_epsilon - min value of the epsilon parameter
    # decay_rate - exponential decay rate for exploration prob
    # init_state - the initial state to start each episode from
    # return: the Qtable learned by the algorithm
    #

    # ====== YOUR CODE: ======
    state_size = mdp.num_row * mdp.num_col
    actions = list(mdp.actions)
    qtable = np.zeros((state_size, len(actions)))

    for s in mdp.terminal_states:
        for action_index in range(len(actions)):
            qtable[one_dim_conversion(s, mdp.num_col), action_index] = float(mdp.board[s[0]][s[1]])

    for episode in range(total_episodes):
        # Reset the environment
        state = init_state
        done = False

        for step in range(max_steps):
            # Choose an action (a) in the current world state (s)
            # First we randomize a number
            exp_exp_tradeoff = random.uniform(0, 1)

            # If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state
            if exp_exp_tradeoff > epsilon:
                action = actions[np.argmax(qtable[one_dim_conversion(state, mdp.num_col), :])]

            # Else doing a random choice --> exploration
            else:
                non_weighted_action = random.choice(actions)  # we consider the randomness of the transition function
                action = random.choices(population=actions, weights=mdp.transition_function[non_weighted_action], k=1)[0]

            # Take the action (a) and observe the outcome state (s') and reward (r)
            new_state = mdp.step(state, action)
            action_index = actions.index(action)
            reward = float(mdp.board[state[0]][state[1]])
            done = (new_state in mdp.terminal_states)

            # Update Q(s,a):=Q(s,a) + Lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            qtable[one_dim_conversion(state, mdp.num_col), action_index] += learning_rate * (reward + mdp.gamma * np.max(qtable[one_dim_conversion(new_state, mdp.num_col), :]) -
                                                                            qtable[one_dim_conversion(state, mdp.num_col), action_index])
            # Our new state is state
            state = new_state

            # If done : finish episode
            if done == True:
                break

        # Reduce epsilon (because we need less and less exploration
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    return qtable
    # ========================


def reshape(list1, list2):
    last = 0
    res = []
    for ele in list1:
        res.append(list2[last: last + len(ele)])
        last += len(ele)

    return res


def q_table_policy_extraction(mdp, qtable):
    # TODO:
    # Given the mdp and the Qtable:
    # return: the policy corresponding to the Qtable
    #

    # ====== YOUR CODE: ======
    policy = []
    actions = list(mdp.actions)
    for state in range(qtable.shape[0]):
        policy.append(actions[np.random.choice(np.flatnonzero(np.isclose(qtable[state, :], qtable[state, :].max())))])
        # policy.append(actions[np.argmax(qtable[state, :])])
    return reshape(mdp.board, policy)
    # ========================


# BONUS

def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
