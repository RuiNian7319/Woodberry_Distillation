"""
Wood-Berry Distillation Column Simulation with Reinforcement Learning for Fault Tolerant Control

By: Rui Nian

Date of Last Edit: Feb 13th 2019


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
import seaborn as sns
import random

from copy import deepcopy
from scipy.integrate import odeint

import gc

import warnings
import sys

sys.path.insert(0, '/home/rui/Documents/IOL_Fault_Tolerant_Control/Woodberry_Distillation')
sys.path.insert(0, '/Users/ruinian/Documents/MATLAB/Woodberry_Distillation')

from RL_Module_Velocity_SMDP import ReinforceLearning


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

        # Setpoint changes
        self.set_point = np.zeros(nsim + 1)

    def ode(self, state, t, inputs):
        """
        Description
             -----
             MIMO state space model of the Wood-Berry Distillation Tower.  Contains 4 states and 2 actions.  The dxdts
             may be able to be optimized through dot product?


        Inputs
             -----
                state: States of the system at time t - 1. Current states has no physical meaning. [x1, x2, x3, x4]
                    t: Limits of integration for sp.odeint.  [t - 1, t]
               inputs: Control inputs into the ordinary differential equations. [u1, u2]


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

    def step(self, inputs, time, setpoint, noise=False, economics='distillate'):
        """
        Description
             -----



        Inputs
             -----



        Returns
             -----

        """

        self.set_point[time] = setpoint

        # Account for delay of the models
        delay_u = np.array([self.u[time - 1, 0], self.u[time - 7, 0], self.u[time - 3, 1], self.u[time - 3, 1]])

        # Integrate the states to calculate for the next states
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

        new_state = deepcopy(self.y[time, :])

        if time == (self.Nsim - 1):
            done = True
        else:
            done = False

        reward = self.reward_calculator(setpoint, time, economics=economics)

        info = "placeholder"

        return new_state, reward, done, info

    def reward_calculator(self, setpoint, time, economics='distillate'):
        """
        Description
             -----



        Inputs
             -----



        Returns
             -----

        """

        if economics == 'distillate':
            error_y1 = abs(self.y[time, 0] - setpoint)
            reward = -error_y1

        elif economics == 'bottoms':
            error_y2 = abs(self.y[time, 1] - setpoint)
            reward = -error_y2

        elif economics == 'all':
            error_y1 = abs(self.y[time, 0] - setpoint[0])
            error_y2 = abs(self.y[time, 1] - setpoint[1])
            reward = -(error_y1 + error_y2)

        else:
            raise ValueError('Improper type selected')

        return reward

    def actuator_fault(self, actuator_num, actuator_value, time, noise=False):
        """
        Description
             -----



        Inputs
             -----



        Returns
             -----

        """

        # If actuator 1 is selected
        if actuator_num == 1:
            self.u[time - 1, 0] = actuator_value

            # If noise is enabled for actuator 1
            if noise:
                self.u[time - 1, 0] += np.random.normal(0, 0.3)

        # If actuator 2 is selected
        if actuator_num == 2:
            self.u[time - 1, 1] = actuator_value

            # If noise is enabled for actuator 2
            if noise:
                self.u[time - 1, 1] += np.random.normal(0, 0.3)

    def sensor_fault(self, sensor_num, sensor_value):
        """
        Description
             -----
                Currently a dummy placeholder


        Inputs
             -----



        Returns
             -----

        """

        if sensor_num == 1:
            self.u = self.u
            pass

        if sensor_num == 2:
            pass

        return sensor_value

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
        self.y = np.zeros((self.Nsim + 1, 2))

        self.x = np.zeros((self.Nsim + 1, 4))
        self.u = np.zeros((self.Nsim + 1, 2))

        # Populate the initial states, if rand_init, add white noise sampled from uniform distribution.
        if rand_init:
            self.x[:] = self.x0 + np.random.uniform(-20, 20, size=(1, 2))
            self.u[:] = self.u0 + np.random.uniform(-3, 3, size=(1, 2))
        else:
            self.x[:] = self.x0
            self.u[:] = self.u0

        self.y[:, 0] = self.C[0, 0] * self.x[0, 0]
        self.y[:, 1] = self.C[1, 1] * self.x[0, 1]

        # Setpoint changes
        self.set_point = np.zeros((self.Nsim + 1, 1))

    def plots(self, timestart=50, timestop=550):
        """
        Description
             -----
                Plots the %MeOH in the distillate and bottoms as a function of time.


        Inputs
             -----
                timestart: What time (in simulation time) to start plotting
                 timestop: What time (in simulation time) to stop plotting

        """

        plt.plot(self.timestep[timestart:timestop], self.y[timestart:timestop, 0], label='$X_D$')
        plt.plot(self.timestep[timestart:timestop], self.y[timestart:timestop, 1], label='$X_B$')

        plt.xlabel(r'Time, \textit{t} (s)')
        plt.ylabel(r'\%MeOH, \textit{X} (wt. \%)')

        plt.legend(loc=0, prop={'size': 12}, frameon=False)

        plt.show()

    def cost_function(self, output='distillate', error_type='ISE', dead_period=15):
        """
        Description
             -----



        Inputs
             -----
                error:
          dead_period:

        Returns
             -----
                error:

        """

        error = 0

        # Integral of absolute error evaluation
        if error_type == "IAE":
            if output == 'distillate':
                error = abs(self.y[dead_period:, 0].reshape(-1, 1) - self.set_point[dead_period:])
                error = sum(error) / (self.Nsim - dead_period)
            elif output == 'bottoms':
                error = abs(self.y[dead_period:, 1].reshape(-1, 1) - self.set_point[dead_period:])
                error = sum(error) / (self.Nsim - dead_period)

        # Integral of squared error evaluation
        elif error_type == "ISE":
            if output == 'distillate':
                error = np.power(self.y[dead_period:, 0].reshape(-1, 1) - self.set_point[dead_period:], 2)
                error = sum(error) / (self.Nsim - dead_period)
            elif output == 'bottoms':
                error = np.power(self.y[dead_period:, 1].reshape(-1, 1) - self.set_point[dead_period:], 2)
                error = sum(error) / (self.Nsim - dead_period)

        else:
            raise ValueError('Improper error evaluation selected.')

        return error


class DiscretePIDControl:
    """


    """
    def __repr__(self):
        return "DiscretePIDControl({}, {}, {})".format(self.Kp, self.Ki, self.Kd)

    def __str__(self):
        return "Discrete-Time PID Controller"

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
        self.error = []

    def __call__(self, setpoint, x_cur, x_1, x_2, eval_time=4):
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

        self.error.append(ek)

        du = self.Kp * (ek - ek_1) + self.Ki * ek + self.Kd * (ek - 2 * ek_1 + ek_2)

        # Constraints on output of PID
        # control_action = max(0, min(last_u + du, 20))
        control_action = self.u[-1] + du

        # Used to synchronize PID inputs with plant outputs if plant and PID are evaluated at different time periods
        for _ in range(eval_time):
            self.u.append(control_action)

        return control_action

    def reset(self):
        """
        Description
             -----
                Resets the PID input trajectory.

        """
        self.u = []


if __name__ == "__main__":

    # Build RL Objects
    rl = ReinforceLearning(discount_factor=0.95, states_start=300, states_stop=340, states_interval=0.5,
                           actions_start=-15, actions_stop=15, actions_interval=2.5, learning_rate=0.5,
                           epsilon=0.2, doe=1.2, eval_period=30)

    # Building states for the problem, states will be the tracking errors
    states = np.linspace(-30, 10, 201)

    rl.user_states(list(states))

    # Building actions for the problem, actions will be inputs of u2
    actions = np.linspace(-15, 15, 121)

    rl.user_actions(actions)

    # Load Q, T, and NT matrices from previous training
    # q = np.loadtxt("Q_Matrix.txt")
    # t = np.loadtxt("T_Matrix.txt")
    # nt = np.loadtxt("NT_Matrix.txt")
    #
    # rl.user_matrices(q, t, nt)
    # del q, t, nt, actions

    # Build PID Objects
    PID1 = DiscretePIDControl(kp=1.31, ki=0.21, kd=0)
    PID2 = DiscretePIDControl(kp=-0.28, ki=-0.06, kd=0)

    # Set initial conditions
    PID1.u = [3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9]
    PID2.u = [0, 0, 0, 0, 0, 0, 0, 0]

    init_state = np.array([65.13, 42.55, 0.0, 0.0])
    init_input = np.array([3.9, 0.0])

    env = WoodBerryDistillation(nsim=6000, x0=init_state, u0=init_input)

    # Starting at time 7 because the largest delay is 7
    input_1 = env.u[0, 0]
    input_2 = env.u[0, 1]
    set_point1 = 100
    set_point2 = 0

    episodes = 3001
    rlist = []

    for episode in range(episodes):

        # Resetting environment and PID controllers
        env.reset(rand_init=False)
        PID1.u = [3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9]
        PID2.u = [0, 0, 0, 0, 0, 0, 0, 0]

        input_1 = env.u[0, 0]
        input_2 = env.u[0, 1]

        tot_reward = []
        state = 0
        action = set_point2
        action_index = 0
        action_list = [set_point2]
        time_list = [0]

        tracker = 0

        # Fault Detection
        # deltaU = []
        # deltaY = []

        # SMDP Reward tracking
        cumu_reward = []

        # Valve stuck position
        if episode % 10 == 0:
            valve_pos = 12
        else:
            valve_pos = np.random.uniform(7, 13.5)

        """
        Loop Description
           ---
              Loop over one episode
        """
        for t in range(7, env.Nsim + 1):

            tau = rl.eval_period

            # PID Evaluate
            if t % 4 == 0 and t < 170:
                input_1 = PID1(set_point1, env.y[t - 1, 0], env.y[t - 2, 0], env.y[t - 3, 0])
                input_2 = PID2(set_point2, env.y[t - 1, 1], env.y[t - 2, 1], env.y[t - 3, 1])

            # Set-point change
            # if t == 100:
            #     set_point1 = 65
            #     set_point2 += 2

            # Disturbance
            # if 350 < t < 370:
            #     env.x[t - 1, :] = env.x[t - 1, :] + np.random.normal(0, 3, size=(1, 4))

            # Actuator Faults
            if 105 < t:
                env.actuator_fault(actuator_num=1, actuator_value=valve_pos, time=t, noise=False)

            # RL Controls
            if 150 < t:
                if t % rl.eval_period == 0 or rl.next_eval:

                    rl.next_eval = False

                    tracker += 1

                    # RL evaluation time
                    rl.eval = t

                    state, action, action_index = rl.action_selection(env.y[t - 1, 0] - set_point1, action_list[-1],
                                                                      no_decay=25, ep_greedy=True, time=t,
                                                                      min_eps_rate=0.01)
                    # To see how well the PID is tracking RL
                    action_list.append(action)
                    time_list.append(t)

            if 170 < t and t % 4 == 0:
                input_2 = PID2(action, env.y[t - 1, 1], env.y[t - 2, 1], env.y[t - 3, 1])

            # Generate input tuple
            control_input = np.array([[input_1, input_2]])

            # Simulate next time
            next_state, Reward, Done, Info = env.step(control_input, t, setpoint=set_point1, noise=False,
                                                      economics='distillate')

            # Reached steady state given by RL or system did not reach the state in 30 seconds,
            if next_state[1] * 0.997 < action < next_state[1] * 1.003 and t - rl.eval > 15:
                rl.eval_feedback = t
                tau = t - rl.eval

            # Append cumulative reward
            cumu_reward.append(Reward)

            # For fault detection
            # if t % 5 == 0 and t != 0:
            #     deltaU.append(abs(PID1.u[t] - PID1.u[t - 10]))
            #     deltaY.append(abs(env.y[t, 0] - env.y[t - 5, 0]))

            # RL Feedback
            if t == rl.eval_feedback and t > 150:

                # Calculate and reset cumulative reward
                reward_rate = np.average(cumu_reward)
                cumu_reward = []

                # Update RL Matrices
                rl.matrix_update(action_index, reward_rate, state, env.y[t, 0] - set_point1, 5, tau)
                tot_reward.append(reward_rate)

                # Define eval period for next state
                rl.next_eval = True

        rlist.append(np.average(tot_reward))

        # Autosave Q, T, and NT matrices
        rl.autosave(episode, 100)

        if episode % 10 == 0:
            print("Episode {} | Episode Reward {}".format(episode, np.average(tot_reward)))

    env.plots(timestart=50, timestop=6000)

    # plt.scatter(PID1.u[40:env.y.shape[0]], env.y[40:, 0])
    # plt.show()
    #
    # plt.scatter(PID2.u[40:env.y.shape[0]], env.y[40:, 1])
    # plt.show()
