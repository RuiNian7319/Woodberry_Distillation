"""
Wood-Berry Distillation Column Simulation (Decoupled).
This simulator broke the TITO Wood-Berry distillation column to 2 distributed SISO systems.

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
               Nsim:  Length of simulation
                 x0:  Initial conditions for states, x ~ X
                 u0:  Initial conditions for inputs, u ~ U
                 xs:  Optimal steady state states, x_s
                 us:  Optimal steady state inputs, u_s
          step_size:  Size of each step for integration purposes, 1 represents 1 second in simulation time
                  y:  Outputs of the system at different time steps, [X_D, X_B, Water_D, Water, B]
                  x:  States of the system at different time steps
                  u:  Inputs to the system at different time steps
                  A:  System matrix
                  B:  Input matrix
                  C:  Output matrix
                  D:  Feedforward matrix
           timestep:  Sequential time steps for the whole simulation


    Methods
         -----
                ode:  Ordinary differential equations of the system.  Contains 4 states and 2 inputs
               step:  Simulates one step of the simulation using odeint from Scipy
              reset:  Reset current simulation



    """

    def __repr__(self):
        return "WoodBerryDistillation({}, {}, {})".format(self.nsim, self.x0, self.u0)

    def __str__(self):
        return "Wood-Berry distillation simulation object."

    def __init__(self, nsim, x0, u0, xs=np.array([2.6219, 1.7129, 1.113, 0.7632]), us=np.array([0.157, 0.05337]),
                 step_size=1):
        """
        Description
             -----



        Variables
             -----

        """
        self.Nsim = nsim
        self.xs = xs
        self.us = us
        self.step_size = step_size

        # State space model
        self.A = np.array([[-0.07699, 0], [0, -0.08929]])
        self.B = np.array([[0.5, 0], [0, 1]])
        self.C = np.array([[0.9809, 0], [0, -0.8621]])
        self.D = 0

        # Output, state, and input trajectories
        self.y = np.zeros((nsim + 1, 4))

        self.x = np.zeros((nsim + 1, 2))
        self.u = np.zeros((nsim + 1, 2))

        # Populate the initial states
        self.x[:] = x0
        self.u[:] = u0

        self.y[:, 0] = self.C[0, 0] * self.x[0, 0]
        self.y[:, 1] = self.C[1, 1] * self.x[0, 1]
        self.y[:, 2] = 100 - self.y[0, 0]
        self.y[:, 3] = 100 - self.y[0, 1]

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

        u1 = inputs[0]
        u2 = inputs[1]

        dxdt1 = self.A[0, 0] * x1 + self.B[0, 0] * u1
        dxdt2 = self.A[1, 1] * x2 + self.B[1, 1] * u2

        dxdt = [dxdt1, dxdt2]

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

        delay_u = np.array([self.u[time - 1, 0], self.u[time - 2, 1]])

        x_next = odeint(self.ode, self.x[time - 1], [self.timestep[time - 1], self.timestep[time]], args=(delay_u, ))

        # odeint outputs the current time and the last time's x, so x_next[-1] is taken.
        # State, input, and output trajectories
        self.x[time, :] = x_next[-1]
        self.u[time, :] = inputs[0]

        self.y[time, 0] = self.C[0, 0] * self.x[time, 0]
        self.y[time, 1] = self.C[1, 1] * self.x[time, 1]

        # Ensure compositions are always between 0 and 100
        # for i, comp in enumerate(self.y[time, :]):
        #     if comp > 100:
        #         self.y[time, i] = 100
        #     elif comp < 0:
        #         self.y[time, i] = 0
        #     else:
        #         pass

        # Add compositions for water
        self.y[time, 2] = 100 - self.y[time, 0]
        self.y[time, 3] = 100 - self.y[time, 1]

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

    def plots(self):

        plt.plot(self.y[:, 0])
        plt.plot(self.y[:, 1])
        plt.axhline(y=0, color='red')
        plt.axhline(y=100, color='red')
        plt.show()


class PIDControl:
    """


    """
    def __repr__(self):
        return "PIDControl({}, {}, {})".format(self.Kp, self.Ki, self.Kd)

    def __str__(self):
        return "PID Controller"

    def __init__(self, kp, ki, kd):
        """
        Descriptions
             -----
                Class for a discrete Proportional-Integral-Derivative Controller.
                Original form: du = Kp * (ek - ek_1) + Kp * h / Ti * ek + Kp * Td / h * (ek - 2 * ek_1 + ek_2)

                Modifications: Ki = Kp * h / Ti
                               Kd = Kp ( Td / h)

                     New form: du = Kp * (ek - ek_1) + Ki * ek + Kd * (ek - 2 * ek_1 + ek_2)

        Attributes
             -----
                kp:  Controller proportional gain
                ki:  Controller integral gain
                kd:  Controller derivative gain

        """

        # Controller parameters
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd

        # Process parameters
        self.error = []

    def __call__(self, setpoint, x_cur, x_1, x_2, last_u):

        ek = setpoint - x_cur
        ek_1 = setpoint - x_1
        ek_2 = setpoint - x_2

        du = self.Kp * (ek - ek_1) + self.Ki * ek + self.Kd * (ek - 2 * ek_1 + ek_2)

        # Append controller error
        self.error.append(ek)

        return last_u + du


if __name__ == "__main__":

    PID1 = PIDControl(kp=1.8, ki=0.21, kd=0)
    PID2 = PIDControl(kp=-0.28, ki=-0.075, kd=0)

    init_state = np.array([51, -58])
    init_input = np.array([0, 0])

    env = WoodBerryDistillation(nsim=600, x0=init_state, u0=init_input)

    # Starting at time 7 because the largest delay is 7
    input_1 = 10
    input_2 = 5
    set_point1 = 100
    set_point2 = 0

    for t in range(3, env.Nsim + 1):

        if t % 3 == 0:
            input_1 = PID1(set_point1, env.y[t - 1, 0], env.y[t - 2, 0], env.y[t - 3, 0], env.u[t - 1, 0])
            input_2 = PID2(set_point2, env.y[t - 1, 1], env.y[t - 2, 1], env.y[t - 3, 1], env.u[t - 1, 1])

        # Set-point change
        if t % 100 == 0:
            set_point1 = 50
            set_point2 = 10

        # Disturbance
        if t % 200 == 0:
            env.x[t - 1, :] = env.x[t - 1, :] + np.random.normal(0, 5, size=(1, 2))

        control_input = np.array([[input_1, input_2]])

        State, Reward, Done, Info = env.step(control_input, t)

    plt.plot(env.y[:, 0])
    plt.plot(env.y[:, 1])
    plt.axhline(y=0, color='red')
    plt.axhline(y=100, color='red')
    plt.show()
