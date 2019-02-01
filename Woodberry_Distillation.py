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
        return "WoodBerryDistillation({}, {}, {})".format(self.nsim, self.x0, self.u0)

    def __str__(self):
        return "Wood-Berry distillation simulation object."

    def __init__(self, nsim, x0, u0, xs=np.array([2.6219, 1.7129, 1.113, 0.7632]), us=np.array([15.7, 5.337]),
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

        # State space model
        self.A = np.array([[-0.0599, 0, 0, 0], [0, -0.0917, 0, 0], [0, 0, -0.0476, 0], [0, 0, 0, -0.0694]])
        self.B = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        self.C = np.array([[0.7665, 0, -0.9, 0], [0, 0.6055, 0, -1.3472]])
        self.D = 0

        # Output, state, and input trajectories
        self.y = np.zeros((nsim + 1, 2))

        self.x = np.zeros((nsim + 1, 4))
        self.u = np.zeros((nsim + 1, 2))

        # Populate the initial states
        self.x[:] = x0
        self.u[:] = u0

        self.y[:, 0] = self.C[0, 0] * self.x[0, 0] + self.C[0, 2] * self.x[0, 2]
        self.y[:, 1] = self.C[1, 1] * self.x[0, 1] + self.C[1, 3] * self.x[0, 3]

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

        u11 = inputs[0]
        u12 = inputs[1]

        u21 = inputs[2]
        u22 = inputs[3]

        dxdt1 = self.A[0, 0] * x1 + self.B[0, 0] * u11
        dxdt2 = self.A[1, 1] * x2 + self.B[1, 0] * u12
        dxdt3 = self.A[2, 2] * x3 + self.B[2, 1] * u21
        dxdt4 = self.A[3, 3] * x4 + self.B[3, 1] * u22

        dxdt = [dxdt1, dxdt2, dxdt3, dxdt4]

        return dxdt

    def step(self, inputs, time, noise=False):
        """
        Description
             -----



        Inputs
             -----



        Returns
             -----

        """

        delay_u = np.array([self.u[time - 1, 0], self.u[time - 7, 0], self.u[time - 3, 1], self.u[time - 3, 1]])

        x_next = odeint(self.ode, self.x[time - 1], [self.timestep[time - 1], self.timestep[time]], args=(delay_u, ))

        # odeint outputs the current time and the last time's x, so x_next[-1] is taken.
        # State, input, and output trajectories
        self.x[time, :] = x_next[-1]
        self.u[time, :] = inputs[0]

        if noise:
            self.y[time, 0] = self.C[0, 0] * self.x[time, 0] + self.C[0, 2] * self.x[time, 2] + np.random.normal(0, 0.2)
            self.y[time, 1] = self.C[1, 1] * self.x[time, 1] + self.C[1, 3] * self.x[time, 3] + np.random.normal(0, 0.2)
        else:
            self.y[time, 0] = self.C[0, 0] * self.x[time, 0] + self.C[0, 2] * self.x[time, 2]
            self.y[time, 1] = self.C[1, 1] * self.x[time, 1] + self.C[1, 3] * self.x[time, 3]

        # Ensure compositions are always between 0 and 100
        # for i, comp in enumerate(self.y[time, :]):
        #     if comp > 100:
        #         self.y[time, i] = 100
        #     elif comp < 0:
        #         self.y[time, i] = 0
        #     else:
        #         pass

        state = deepcopy(self.y[time, :])

        if time == (self.Nsim - 1):
            done = True
        else:
            done = False

        reward = "placeholder"

        info = "placeholder"

        return state, reward, done, info

    def actuator_fault(self, actuator_num, actuator_value, time):
        """
        Description
             -----



        Inputs
             -----



        Returns
             -----

        """

        if actuator_num == 1:
            self.u[time - 1, 0] = actuator_value + np.random.normal(0, 0.2)

        if actuator_num == 2:
            self.u[time - 1, 1] = actuator_value + np.random.normal(0, 0.2)

    def sensor_fault(self, sensor_num, sensor_value):
        """
        Description
             -----



        Inputs
             -----



        Returns
             -----

        """

        if actuator_num == 1:
            pass

        if actuator_num == 2:
            pass

    def reset(self, rand_init=False):
        """
        Description
             -----



        Inputs
             -----



        Returns
             -----

        """

        # Output, state, and input trajectories
        self.y = np.zeros((self.Nsim + 1, 4))

        self.x = np.zeros((self.Nsim + 1, 2))
        self.u = np.zeros((self.Nsim + 1, 2))

        # Populate the initial states
        if rand_init:
            self.x[:] = self.x0 + np.random.uniform(-20, 20, size=(1, 2))
            self.u[:] = self.u0 + np.random.uniform(-3, 3, size=(1, 2))
        else:
            self.x[:] = self.x0
            self.u[:] = self.u0

        self.y[:, 0] = self.C[0, 0] * self.x[0, 0]
        self.y[:, 1] = self.C[1, 1] * self.x[0, 1]
        self.y[:, 2] = 100 - self.y[0, 0]
        self.y[:, 3] = 100 - self.y[0, 1]

    def plots(self, timestart=50, timestop=550):
        """
        Description
             -----



        Inputs
             -----



        Returns
             -----

        """

        plt.plot(self.y[timestart:timestop, 0], label='$X_D$')
        plt.plot(self.y[timestart:timestop, 1], label='$X_B$')

        plt.xlabel(r'Time, \textit{t} (s)')
        plt.ylabel(r'\%MeOH, \textit{X} (wt. \%)')

        plt.legend(loc=0, prop={'size': 12}, frameon=False)

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

        # Controls from the digital controller
        self.u = []

    def __call__(self, setpoint, x_cur, x_1, x_2, last_u, eval_time=4):
        """
        Description
             -----



        Inputs
             -----



        Returns
             -----

        """

        ek = setpoint - x_cur
        ek_1 = setpoint - x_1
        ek_2 = setpoint - x_2

        du = self.Kp * (ek - ek_1) + self.Ki * ek + self.Kd * (ek - 2 * ek_1 + ek_2)

        # Constraints on output of PID
        # control_action = max(0, min(last_u + du, 20))
        control_action = last_u + du

        # Used to synchronize PID inputs with plant outputs if plant and PID are evaluated at different time periods
        for _ in range(eval_time):
            self.u.append(control_action)

        return control_action


if __name__ == "__main__":

    # Build PID Objects
    PID1 = PIDControl(kp=1.31, ki=0.21, kd=0)
    PID2 = PIDControl(kp=-0.28, ki=-0.06, kd=0)

    # Set initial conditions
    PID1.u = [3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9]
    PID2.u = [0, 0, 0, 0, 0, 0, 0, 0]

    init_state = np.array([65.13, 42.55, 0.0, 0.0])
    init_input = np.array([3.9, 0.0])

    env = WoodBerryDistillation(nsim=550, x0=init_state, u0=init_input)

    # Starting at time 7 because the largest delay is 7
    input_1 = env.u[0, 0]
    input_2 = env.u[0, 1]
    set_point1 = 100
    set_point2 = 0

    for t in range(7, env.Nsim + 1):

        if t % 10000 == 0:
            print(t)

        if t % 4 == 0:
            input_1 = PID1(set_point1, env.y[t - 1, 0], env.y[t - 2, 0], env.y[t - 3, 0], env.u[t - 1, 0])
            # input_2 = PID2(set_point2, env.y[t - 1, 1], env.y[t - 2, 1], env.y[t - 3, 1], env.u[t - 1, 1])

        # Set-point change
        if t == 50:
            set_point1 = 60
            # set_point2 += 2

        # Disturbance
        # if t % 320 == 0:
        #     env.x[t - 1, :] = env.x[t - 1, :] + np.random.normal(0, 5, size=(1, 4))

        # Actuator Faults
        if 55 < t:
            env.actuator_fault(actuator_num=1, actuator_value=13, time=t)

        # Generate input tuple
        control_input = np.array([[input_1, input_2]])

        # Simulate next time
        State, Reward, Done, Info = env.step(control_input, t, noise=False)

    env.plots()
    # plt.scatter(PID1.u[40:env.y.shape[0]], env.y[40:, 0])
    # plt.show()

    # plt.scatter(PID2.u[40:env.y.shape[0]], env.y[40:, 1])
    # plt.show()
