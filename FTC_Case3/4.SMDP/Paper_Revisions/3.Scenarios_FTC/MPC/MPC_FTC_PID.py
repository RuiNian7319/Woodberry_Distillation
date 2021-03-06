import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '/home/rui/Documents/Research/Models')
sys.path.insert(0, '/home/rui/Documents/Research/Modules')

from copy import deepcopy
from Woodberry_CasaDI import WoodberryDistillation
from MPC_Module_Discounted import ModelPredictiveControl


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


def simulation():

    # Build PID Objects
    PID1 = DiscretePIDControl(kp=1.31, ki=0.21, kd=0)
    PID2 = DiscretePIDControl(kp=-0.28, ki=-0.06, kd=0)

    # Initialize PIDs
    PID1.u = [3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9]
    PID2.u = [0, 0, 0, 0, 0, 0, 0, 0]

    input_1 = PID1.u[-1]
    input_2 = PID2.u[-1]

    # Initial set-points
    set_point1 = 100
    set_point2 = 0

    # Plant model
    model_plant = WoodberryDistillation(nsim=2000)

    # Build Controller Model
    model_control = WoodberryDistillation(nsim=model_plant.Nsim,
                                          x0=np.array([65.13, 42.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                          xs=np.array([261.78, 171.18, 111.43, 76.62, 0.0, 0.0, 0.0, 0.0]),
                                          control=True)

    """
    Below are all the MPC and controller objects
    """

    # Build MPC Object
    control = ModelPredictiveControl(model_control.Nsim, 10, model_control.Nx, model_control.Nu, 1, 0.0, 0.0,
                                     model_control.xs, model_control.us, eval_time=15, dist=True, gamma=0.9,
                                     upp_u_const=[99, 99, 99, 99], low_u_const=[0.0, 0.0, 0.0, 0.0],
                                     upp_x_const=[1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
                                     low_x_const=[-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000])

    # MPC Construction
    mpc_control = control.get_mpc_controller(model_control.system_ode, delta=control.eval_time, x0=model_control.x0,
                                             verbosity=0, random_guess=False)

    # Build FTC-MPC Object
    ftc_control = ModelPredictiveControl(model_control.Nsim, 10, model_control.Nx, model_control.Nu, 1, 0.0, 0.0,
                                         np.array([200.33, 130.86, 60.87, 41.74, 0, 0, 0, 0]), model_control.us,
                                         eval_time=15, dist=True, gamma=0.9,
                                         upp_u_const=[12.1, 12.1, 99, 99], low_u_const=[11.9, 11.9, 0, 0],
                                         upp_x_const=[1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
                                         low_x_const=[-1000, -1000, -1000, -1000, -1000, -1000, -1000, -1000])

    # MPC FTCConstruction
    ftc_mpc_control = ftc_control.get_mpc_controller(model_control.system_ode, delta=control.eval_time,
                                                     x0=model_control.x0,
                                                     verbosity=0, random_guess=False)

    for t in range(7, model_plant.Nsim + 1):

        # Initial 355 minutes of simulation
        if t % 15 == 0 and t <= 355:
            # Solve the MPC optimization problem, obtain current input and predicted state
            model_control.u[t, :], model_control.x[t, :] = control.solve_mpc(model_plant.x, model_plant.xsp,
                                                                             mpc_control, t, control.p)

        # Evaluate FT-MPC
        elif t % 15 == 0 and t > 355:
            # Solve the MPC optimization problem, obtain current input and predicted state
            model_control.u[t, :], model_control.x[t, :] = ftc_control.solve_mpc(model_plant.x, model_plant.xsp,
                                                                                 ftc_mpc_control, t, control.p)
            print(t, model_control.u[t, :])

        # When MPC does not evaluate
        else:
            model_control.u[t, :] = model_control.u[t - 1, :]
            model_control.x[t, :] = model_control.x[t - 1, :]

        # Fault in the system
        if t >= 340:
            model_control.u[t, 0:2] = 12

        if t % 4 == 0:
            input_1 = PID1(model_control.u[t, 0], env.y[t - 1, 0], env.y[t - 2, 0], env.y[t - 3, 0])
            input_2 = PID2(model_control.u[t, 2], env.y[t - 1, 1], env.y[t - 2, 1], env.y[t - 3, 1])

        # Generate input tuple
        control_input = np.array([input_1, input_2])

        # Calculate the next states for the plant
        model_plant.x[t, :] = model_plant.system_sim.sim(model_plant.x[t - 1, :], control_input)

        # Populate the output values for model and control
        model_control.y[t, 0] = model_control.C[0, 0]*model_control.x[t, 0]+model_control.C[0, 2]*model_control.x[t, 2]
        model_control.y[t, 1] = model_control.C[1, 1]*model_control.x[t, 1]+model_control.C[1, 3]*model_control.x[t, 3]

        model_plant.y[t, 0] = model_plant.C[0, 0] * model_plant.x[t, 0] + model_plant.C[0, 2] * model_plant.x[t, 2]
        model_plant.y[t, 1] = model_plant.C[1, 1] * model_plant.x[t, 1] + model_plant.C[1, 3] * model_plant.x[t, 3]

        # Disturbance
        # if t % 20 == 0:
        #     model_plant.x[t, 1] -= 5

        # Update the P parameters for offset-free control
        control.p = model_plant.x[t, :] - model_control.x[t, 0:model_plant.Nx]
        ftc_control.p = model_plant.x[t, :] - model_control.x[t, 0:model_plant.Nx]

        # Update the plant inputs to be same as controller inputs
        if t == model_plant.Nsim:
            model_plant.u = deepcopy(model_control.u)

    return model_plant, model_control, control


if __name__ == "__main__":
    Model_Plant, Model_Control, Controller = simulation()

    # Tests for validation
    # assert(np.allclose(Model_Plant.x, Model_Control.x[:, 0:Model_Plant.Nx]))
