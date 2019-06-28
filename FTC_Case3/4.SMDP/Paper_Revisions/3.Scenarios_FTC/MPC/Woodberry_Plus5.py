import numpy as np
import matplotlib.pyplot as plt
import random
import mpctools as mpc

from Box import Box
from copy import deepcopy


class WoodberryDistillation:

    def __init__(self, nsim, x0=np.array([65.13, 42.55, 0.0, 0.0]), u0=np.array([3.9, 3.9, 0.0, 0.0]),
                 xs=np.array([261.78, 171.18, 111.43, 76.62]), us=np.array([15.7, 15.7, 5.337, 5.337]),
                 step_size=1, control=False, q_cost=1, r_cost=0.5, random_seed=1):

        # Initial conditions and other required parameters
        self.Nsim = nsim
        self.x0 = x0
        self.u0 = u0
        self.xs = xs
        self.us = us
        self.step_size = step_size
        self.t = np.linspace(0, nsim * self.step_size, nsim + 1)
        self.control = control

        # Model Parameters
        if self.control:
            self.Nx = 8
        else:
            self.Nx = 4

        # Double Nu to account for time delay
        self.Nu = int(self.u0.shape[0])
        self.action_space = Box(low=np.array([-5]), high=np.array([5]))
        self.observation_space = np.zeros(self.Nx)
        self.Q = q_cost * np.eye(self.Nx)
        self.R = r_cost * np.eye(self.Nu)

        # State space model
        self.A = np.array([[-0.0629, 0, 0, 0], [0, -0.0963, 0, 0], [0, 0, -0.05, 0], [0, 0, 0, -0.0729]])
        self.B = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        self.C = np.array([[0.7665, 0, -0.9, 0], [0, 0.6055, 0, -1.3472]])
        self.D = 0

        # State and input trajectories
        self.x = np.zeros([nsim + 1, self.Nx])
        self.u = np.zeros([nsim + 1, int(self.Nu)])
        self.x[0:self.Nx, :] = x0
        self.u[0:self.Nx, :] = u0

        # Set initial conditions for the first

        # Output trajectory
        self.y = np.zeros([nsim + 1, 2])

        # Build the CasaDI functions
        self.system_sim = mpc.DiscreteSimulator(self.ode, self.step_size, [self.Nx, self.Nu], ["x", "u"])
        self.system_ode = mpc.getCasadiFunc(self.ode, [self.Nx, self.Nu], ["x", "u"], funcname="odef")

        # Set-point trajectories
        self.xsp = np.zeros([self.Nsim + 1, self.Nx])
        self.xsp[0, :] = self.xs
        self.usp = np.zeros([self.Nsim + 1, int(self.Nu)])
        self.usp[0, :] = self.us

        # Seed the system for reproducability
        random.seed(random_seed)
        np.random.seed(random_seed)

    def ode(self, state, inputs):
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

        # For offset free control
        if self.control:
            dxdt.append(np.zeros(int(self.Nx * 0.5)))

        return dxdt

    def step(self, inputs, step, obj_function="RL"):

        x_next = self.system_sim.sim(self.x[step - 1, :], inputs)

        self.x[step, :] = x_next
        self.u[step, :] = inputs

        self.y[step, 0] = self.C[0, 0] * self.x[step, 0] + self.C[0, 2] * self.x[step, 2]
        self.y[step, 1] = self.C[1, 1] * self.x[step, 1] + self.C[1, 3] * self.x[step, 3]

        state = deepcopy(self.y[step, :])

        reward = self.reward_function(step, obj_function=obj_function)

        if step == (self.Nsim - 1):
            done = True
        else:
            done = False

        info = "placeholder"

        return state, reward, done, info

    def reward_function(self, step, obj_function='RL'):

        # RL Reward function
        if obj_function == "RL":

            reward = 0

            # Set-point tracking error
            if 0.99 * self.xs[0] < self.x[step, 0] < 1.01 * self.xs[0]:
                reward += 0.5 - 0.35 * abs(self.x[step, 0] - self.xs[0])
            else:
                reward += - 0.35 * abs(self.x[step, 0] - self.xs[0])

            # Control input error on u1
            if 0.99 * self.xs[1] < self.x[step, 1] < 1.01 * self.xs[1]:
                reward += 0.5 - 0.35 * abs(self.x[step, 1] - self.xs[1])
            else:
                reward -= 0.35 * abs(self.x[step, 1] - self.xs[1])

            # Control input error on u2
            if 0.99 * self.us[0] < self.u[step, 0] < 1.01 * self.us[0]:
                reward += 0.25 - 0.5 * abs(self.u[step, 0] - self.us[0])
            else:
                reward -= 0.5 * abs(self.u[step, 0] - self.us[0])

        # MPC Reward function
        elif obj_function == "MPC":

            x = self.x[step] - self.xs
            u = (self.u[step, :] - self.us)[0]

            reward = - 0.55 * (x @ self.Q @ x + u * self.R * u) + 1

            reward = max(-10, reward)

        # Improper Reward function
        else:
            raise ValueError("Improper model type specified.")

        return reward

    def __repr__(self):
        return "WoodberryDistillation({}, {}, {}, {}, {})".format(self.Nsim, self.x0, self.u0, self.xs, self.us,
                                                                  self.step_size)

    def __str__(self):
        return "Woodberry distillation with {} time steps".format(self.Nsim)

    @staticmethod
    def seed(number):
        random.seed(number)
        np.random.seed(number)


if __name__ == "__main__":

    env = WoodberryDistillation(nsim=1000, x0=np.array([65.13, 42.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                xs=np.array([261.78, 171.18, 111.43, 76.62, 0.0, 0.0, 0.0, 0.0]), control=True)

    # env = WoodberryDistillation(nsim=1000, x0=np.array([65.13, 42.55, 0.0, 0.0]),
    #                             xs=np.array([2.6219, 1.7129, 1.113, 0.7632]), control=False)

    for time_step in range(7, env.Nsim + 1):

        if 300 < time_step < 600:
            control_input = np.array([[15.7, 15.7, 5.337, 5.337]])
        elif 600 <= time_step:
            control_input = np.array([[15.7, 15.7, 5.337, 5.337]])
        else:
            control_input = np.array([[15.7, 15.7, 5.337, 5.337]])

        State, Reward, Done, Info = env.step(control_input, time_step)
