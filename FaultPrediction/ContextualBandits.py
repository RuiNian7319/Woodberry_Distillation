"""
Contextual Bandits Code

Q(n + 1) = Q(n) + a(R(n) + Q(n)), where Q is f(x, u)
Bias correction for constant step size: B(n) = a / o(n), where o(n) = o(n - 1) + a(1 - o(n - 1)), with o(0) = 0


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
        self.epsilon = epsilon
        self.lr = lr

        # State-Action-Value numbers
        self.Q = np.zeros((len(self.states), len(self.actions)))
        self.T = np.zeros(self.Q.shape)

        # Memory for s and a for updates
        self.s = None
        self.a = None

    def action_selection(self, state, ep_greedy=False, no_decay=5, min_epsilon=0.001):

        state = self.state_detection(cur_state=state)

        action = self.rargmax(self.Q[state, :])

        if ep_greedy:
            self.epsilon_update(no_decay=no_decay, sa_pair=self.T[state, action], min_eps_rate=min_epsilon)

        # Memory for s and a for updates
        self.s = state
        self.a = action

        return action

    def value_update(self, state, action):
        pass

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

        if type(cur_state) == np.float64 or float:

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
            self.epsilon = self.epsilon_0 / (1 + (sa_pair**(1/12) - 1))

        self.epsilon = max(self.epsilon, min_eps_rate)


if __name__ == "__main__":

    bandit = ContextualBandits(states=[-2, -1, 0, 1, 2], actions=[-1, 0, 1])

    bandit.action_selection(-1.6)



