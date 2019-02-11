"""
Simple Implementation of a GridWorld mouse

By: Rui Nian

Date: Feb 8th 2019

Gridworld:

|  9  |  8  |  7  |
|  6  |  5  |  4  |
|  3  |  2  |  1  |

Start = state 1
Cheese = state 9
Shock = state 5
Water = state 7

"""

import numpy as np


class Gridworld:
    """
    Description
         -----
         An environment of a mouse trapped in a grid world.  The mouse tries to reach the cheese, while drinking some
         water along the way.


    Attributes
         -----
               x0: Initial state of the system
                x: State trajectory
                u: Input trajectory


    Methods
         -----
             step: Take the action from RL
           reward: Generate reward based on state
            reset: Reset environment
    """

    def __str__(self):
        return 'A grid world with a mouse seeking a piece of cheese'

    def __repr__(self):
        return 'Gridworld({})'.format(self.Nsim)

    def __init__(self):
        self.x0 = [1]
        self.x = [1]
        self.u = []

    def step(self, action):
        """
        Description
             -----
             Take the action RL selected.  Returns s(t + 1), reward, done, info.


        Inputs
             -----
                action: Action taken by RL.
                           Action 1: Up
                           Action 2: Down
                           Action 3: Left
                           Action 4: Right


        Returns
             -----
            next_state: The new state the agent is in
                reward: Feedback to the agent regarding the last action
                  done: If the episode is terminated
                  info: Debugging purposes, empty for now
        """

        if self.x[-1] == 1:
            if action == 1:
                next_state = 4
            elif action == 2 or action == 4:
                next_state = 1
            elif action == 3:
                next_state = 2
            else:
                raise ValueError('Improper Action')

        elif self.x[-1] == 2:
            if action == 1:
                next_state = 5
            elif action == 2:
                next_state = 2
            elif action == 3:
                next_state = 3
            elif action == 4:
                next_state = 1
            else:
                raise ValueError('Improper Action')

        elif self.x[-1] == 3:
            if action == 1:
                next_state = 6
            elif action == 2:
                next_state = 3
            elif action == 3:
                next_state = 3
            elif action == 4:
                next_state = 2
            else:
                raise ValueError('Improper Action')

        elif self.x[-1] == 4:
            if action == 1:
                next_state = 7
            elif action == 2:
                next_state = 1
            elif action == 3:
                next_state = 5
            elif action == 4:
                next_state = 4
            else:
                raise ValueError('Improper Action')

        elif self.x[-1] == 5:
            if action == 1:
                next_state = 8
            elif action == 2:
                next_state = 2
            elif action == 3:
                next_state = 6
            elif action == 4:
                next_state = 4
            else:
                raise ValueError('Improper Action')

        elif self.x[-1] == 6:
            if action == 1:
                next_state = 9
            elif action == 2:
                next_state = 3
            elif action == 3:
                next_state = 6
            elif action == 4:
                next_state = 5
            else:
                raise ValueError('Improper Action')

        elif self.x[-1] == 7:
            if action == 1:
                next_state = 7
            elif action == 2:
                next_state = 4
            elif action == 3:
                next_state = 8
            elif action == 4:
                next_state = 7
            else:
                raise ValueError('Improper Action')

        elif self.x[-1] == 8:
            if action == 1:
                next_state = 8
            elif action == 2:
                next_state = 5
            elif action == 3:
                next_state = 9
            elif action == 4:
                next_state = 7
            else:
                raise ValueError('Improper Action')

        elif self.x[-1] == 9:
            if action == 1:
                next_state = 9
            elif action == 2:
                next_state = 6
            elif action == 3:
                next_state = 9
            elif action == 4:
                next_state = 8
            else:
                raise ValueError('Improper Action')

        else:
            raise ValueError('Incorrect State')

        self.x.append(next_state)
        self.u.append(action)

        reward = self.reward_calc(next_state)

        if (next_state == 5) or (next_state == 9):
            done = True
        else:
            done = False

        info = 'placeholder'

        return next_state, reward, done, info

    @staticmethod
    def reward_calc(state):
        """
        Description
             -----
             Reward function for the gridworld


        Inputs
             -----
                state: States of the mouse


        Returns
             -----
               reward: Reward for transitioning to that state.
                         Electric shock at s = 5, -1 reward
                         Water at s = 7, +1 reward
                         Start at s = 1
                         Cheese at s = 9, +10 reward

                         Other actions as no effect on reward
        """

        if state == 5:
            reward = -1

        elif state == 9:
            reward = 10

        elif state == 7:
            reward = 1

        else:
            reward = 0

        return reward

    def reset(self):
        """
        Description
             -----
                Resets the gridworld

        """
        self.x = [1]
        self.u = []


if __name__ == "__main__":

    env = Gridworld()

    episodes = 3

    for episode in range(episodes):

        env.reset()
        Done = False

        while Done is False:
            Next_state, Reward, Done, Info = env.step(np.random.randint(1, 4))
            print(Done, episode, Next_state)
