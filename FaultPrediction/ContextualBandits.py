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

    def __init__(self):
        self.states = states
        self.actions = actions

        # State-Action-Value numbers
        self.Q = np.zeros(len(self.states), len(self.actions))

    def action_selection(self):
        pass

    def value_update(self, state, action):
        pass

    def state_detection(self, cur_state):
        """
        Description
             -----



        Inputs
             -----



        Returns
             -----

        """

        if type(cur_state) == np.float64:

            state = min(self.states, key=lambda x_current: abs(x_current - cur_state))
            state = self.states.index(state)

        else:

            state1 = min(self.x1, key=lambda x: abs(x - cur_state[0]))
            state2 = min(self.x2, key=lambda x: abs(x - cur_state[1]))

            state = self.states.index([state1, state2])

        return state





