"""
Wood-Berry Distillation Column Simulation

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
from scipy.integrate import odeint

import gc

import warnings


class WoodBerryDistillation:
    """
    Attributes
         -----



    Methods
         -----



    """

    def __repr__(self):
        return "WoodBerryDistillation({}, {}, {})".format(self.nsim, self.x0, self.u0)

    def __str__(self):
        return "Wood-Berry distillation simulation object."

    def __init__(self, nsim, x0, u0, xs=np.array([2.6219, 1.7129, 1.113, 0.7632]), us=np.array([0.157, 0.053]),
                 step_size=1):
        """
        Description
             -----



        Variables
             -----

        """
        self.Nsim = nsim
        self.x0 = x0
        self.u0 = u0
        self.xs = xs
        self.us = us
        self.step_size = step_size

        # Output, state, and input trajectories
        self.y = np.zeros((nsim + 1, 2))
        self.x = np.zeros((nsim + 1, 4))
        self.u = np.zeros((nsim + 1, 2))
        self.x[0, :] = x0
        self.u[0, :] = u0

        # State space model
        self.A = np.array([[-0.0599, 0, 0, 0], [0, -0.0917, 0, 0], [0, 0, -0.0476, 0], [0, 0, 0, -0.0694]])
        self.B = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        self.C = np.array([[0.7665, 0, -0.9, 0], [0, 0.6055, 0, -1.3472]])
        self.D = 0

        # Timeline of simulation
        self.timestep = np.linspace(0, self.Nsim * self.step_size, self.Nsim + 1)

    def ode(self, state, t, inputs):
        """
        Description
             -----
             MIMO state space model of the Wood-Berry Distillation Tower.  Contains 4 states and 2 actions.  The dxdts
             may be able to be optimized through dot product?


        Inputs
             -----
                state:
               inputs:


        Returns
             -----
                 dxdt: All the equations of the state space model
        """

        x1 = state[0]
        x2 = state[1]
        x3 = state[2]
        x4 = state[3]

        u11 = inputs[0][0]
        u12 = inputs[0][1]

        u21 = inputs[0][0]
        u22 = inputs[0][1]

        dxdt1 = self.A[0, 0] * x1 + self.B[0, 0] * u11
        dxdt2 = self.A[1, 1] * x2 + self.B[1, 0] * u12
        dxdt3 = self.A[2, 2] * x3 + self.B[2, 1] * u21
        dxdt4 = self.A[3, 3] * x4 + self.B[3, 1] * u22

        dxdt = [dxdt1, dxdt2, dxdt3, dxdt4]

        return dxdt

    def step(self, inputs, time):
        """
        Description
             -----



        Inputs
             -----



        Returns
             -----

        """

        x_next = odeint(self.ode, self.x[time - 1], [self.timestep[time - 1], self.timestep[time]], args=(inputs, ))

        # odeint outputs the current time and the last time's x, so x_next[-1] is taken.
        # State, input, and output trajectories
        self.x[time, :] = x_next[-1]
        self.u[time, :] = inputs[0]

        self.y[time, 0] = self.C[0, 0] * self.x[time, 0] + self.C[0, 2] * self.x[time, 2]
        self.y[time, 1] = self.C[1, 1] * self.x[time, 1] + self.C[1, 3] * self.x[time, 3]

        state = deepcopy(self.y[time, :])

        if time == (self.Nsim - 1):
            done = True
        else:
            done = False

        reward = "placeholder"

        info = "placeholder"

        return state, reward, done, info

    def reset(self):
        pass


if __name__ == "__main__":

    init_state = np.array([0, 0, 0, 0])
    init_input = np.array([0, 0])

    env = WoodBerryDistillation(nsim=150, x0=init_state, u0=init_input)

    # Starting at time 7 because the largest delay is 7
    for time_step in range(7, env.Nsim + 1):

        if 30 < time_step < 60:
            control_input = np.array([[0.157, 0.053]])
        elif 60 <= time_step:
            control_input = np.array([[0.157, 0.053]])
        else:
            control_input = np.array([[0.157, 0.053]])

        State, Reward, Done, Info = env.step(control_input, time_step)
