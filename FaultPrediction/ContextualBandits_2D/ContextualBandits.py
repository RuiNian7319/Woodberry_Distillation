"""
Contextual Bandits Code

Q(n + 1) = Q(n) + a(R(n) + Q(n)), where Q is f(x, u)
Bias correction for constant step size: B(n) = a / o(n), where o(n) = o(n - 1) + a(1 - o(n - 1)), with o(0) = 0

Current set-up is based on position rather than velocity.


By: Rui Nian

Date of Last Edit: January 22nd 2019


The MIT License (MIT)
Copyright © 2019 Rui Nian

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above
copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
"""

import numpy as np
import matplotlib.pyplot as plt
import random

from copy import deepcopy

import gc

import warnings

gc.enable()


class ContextualBandits:
    """
    Attributes
         -----
             states: States of the system
            actions: Actions of the system


    Methods
         -----
                ode:



    """

    # Plotting formats
    fonts = {"family": "serif",
             "weight": "normal",
             "size": "12"}

    plt.rc('font', **fonts)
    plt.rc('text', usetex=True)

    # Random Seeding
    random.seed(1)
    np.random.seed(1)

    def __repr__(self):
        return "ContextualBandits(, , )".format()

    def __str__(self):
        return "Contextual Bandit Agent."

    def __init__(self, states, actions, epsilon=0.5, lr=0.5):
        self.states = states
        self.actions = actions
        self.epsilon0 = epsilon
        self.lr0 = lr

        # State-Action-Value numbers
        self.Q = np.random.uniform(-0.0, 0.0, size=(len(self.states), len(self.actions)))
        self.T = np.zeros(self.Q.shape)

        # Memory for s and a for updates
        self.s = None
        self.a = None

        # Memory for epsilon and lr updates
        self.epsilon = self.epsilon0
        self.lr = self.lr0

        # Multiple states
        self.x1 = None
        self.x2 = None

    def action_selection(self, state, ep_greedy=False, no_decay=5, min_epsilon=0.001):

        state_index = self.state_detection(cur_state=state)

        action_index = self.rargmax(self.Q[state_index, :])

        if ep_greedy:
            self.epsilon_update(no_decay=no_decay, sa_pair=self.T[state_index, action_index], min_eps_rate=min_epsilon)
        else:
            self.epsilon = 0

        num = np.random.uniform(0, 1)
        if num < self.epsilon:
            action_index = np.random.randint(0, len(self.actions))
        else:
            pass

        # Memory for s and a for updates
        self.s = state_index
        self.a = action_index

        # Perform action
        action = self.actions[action_index]

        return action

    def value_update(self, reward, no_decay=5):

        # Learning rate update
        self.lr_update(no_decay=no_decay, sa_pair=self.T[self.s, self.a], min_learn_rate=0.001)

        # Update the state-action-value
        self.Q[self.s, self.a] = self.Q[self.s, self.a] + self.lr * (reward - self.Q[self.s, self.a])

        # Update the amount of times that state-action-value is visited
        self.T[self.s, self.a] += 1

    def state_detection(self, cur_state):
        """
        Description
             -----
                Detects the current state of the system.  Because this is a discretized problem, the continuous state
                must be translated to the closest discrete state.


        Inputs
             -----
                cur_state: The current value of the state



        Returns
             -----
                state: The state index from the Q matrix.

        """

        if type(cur_state) == np.float64 or type(cur_state) == float:

            state = min(self.states, key=lambda x_current: abs(x_current - cur_state))
            state = self.states.index(state)

        else:

            state1 = min(self.x1, key=lambda x: abs(x - cur_state[0]))
            state2 = min(self.x2, key=lambda x: abs(x - cur_state[1]))

            state = self.states.index([state1, state2])

        return state

    @staticmethod
    def rargmax(vector):
        """
        Random argmax

        vector: input of numbers

        return: Index of largest number, breaking ties randomly
        """

        m = np.amax(vector)
        indices = np.nonzero(vector == m)[0]

        return random.choice(indices)

    def epsilon_update(self, no_decay, sa_pair, min_eps_rate=0.001):

        if sa_pair < no_decay:
            pass
        else:
            self.epsilon = self.epsilon0 / (1 + (sa_pair**(1/12) - 1))

        self.epsilon = max(self.epsilon, min_eps_rate)

    def lr_update(self, no_decay, sa_pair, min_learn_rate=0.001):

        # During no decay period
        if sa_pair < no_decay:
            pass

        # Decaying learning rate
        else:
            self.lr = self.lr0 / (sa_pair - no_decay + 1)

        self.lr = max(self.lr, min_learn_rate)

    def autosave(self, sim_time, save_time):
        if sim_time % save_time == 0 and sim_time != 0:
            np.savetxt('Q_matrix.csv', self.Q)
            np.savetxt('T_matrix.csv', self.T)

            print("Saving... at time {}.".format(sim_time))


class NumberGame:

    def __init__(self, nsim):
        self.Nsim = nsim

        # State trajectory
        # self.x = np.zeros((self.Nsim, 2))
        self.x = np.loadtxt('contextual_bandit_test.csv').T

    def step(self, action, t):
        # self.x[t, :] = [np.random.uniform(0, 5), np.random.uniform(2, 5)]
        #
        done = False
        #
        # if t > 20:
        #     self.x[t, :] = [np.random.uniform(0, 3), np.random.uniform(0, 4)]
        #
        #     if action == 1:
        #         done = True

        reward = self.reward_calc(action, t)

        next_state = deepcopy(self.x[t, :])

        return next_state, reward, done

    def reward_calc(self, action, t):

        if t > 1000:
            if action == 0:
                reward = -1
            elif action == 1:
                reward = 1
            else:
                raise ValueError('Improper action')

        else:
            if action == 0:
                reward = 0
            elif action == 1:
                reward = -1
            else:
                raise ValueError('Improper action')

        # # Actuator not stuck
        # if abs(self.x[t, 1]) > 5:
        #     if action == 0:
        #         reward = 0
        #     elif action == 1:
        #         reward = -1
        #     else:
        #         raise ValueError('Improper action')
        #
        # # If actuator didn't move, but input also didn't move...
        # elif abs(self.x[t, 0]) < 4:
        #     if action == 0:
        #         reward = 0
        #     elif action == 1:
        #         reward = -1
        #     else:
        #         raise ValueError('Improper action')
        #
        # else:
        #     if action == 0:
        #         reward = -1
        #     elif action == 1:
        #         reward = 1
        #     else:
        #         raise ValueError('Improper action')

        return reward

    def reset(self):
        # self.x = np.zeros((self.Nsim, 2))
        self.x = np.loadtxt('contextual_bandit_test.csv').T


if __name__ == "__main__":

    States = []

    # Changes of PID inputs and process outputs in absolute values
    States1 = np.linspace(0, 12, 13)
    States2 = np.linspace(0, 9, 4)

    for x1 in States1:
        for x2 in States2:
            States.append([x1, x2])

    Actions = [0, 1]

    # Build agent
    Bandit = ContextualBandits(states=States, actions=Actions)

    # Append a multi-state system onto RL
    Bandit.x1 = States1
    Bandit.x2 = States2

    # Load the Q and T matrices
    Bandit.Q = np.loadtxt('Q_matrix.csv')
    Bandit.T = np.loadtxt('T_matrix.csv')

    # Build Environment
    env = NumberGame(nsim=1198)

    episodes = 1

    for episode in range(1, episodes):

        env.reset()

        for step in range(1, env.Nsim):

            Action = Bandit.action_selection(env.x[step - 1], ep_greedy=True)

            State, Reward, Done = env.step(Action, step)

            Bandit.value_update(Reward)

            if Done:
                break

        Bandit.autosave(episode, 500)
