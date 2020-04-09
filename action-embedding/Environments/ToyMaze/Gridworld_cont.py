from __future__ import print_function
#cython: boundscheck=False, wraparound=False, initializedcheck=False

import numpy as np
# cimport numpy as np
import matplotlib.pyplot  as plt
import matplotlib
from matplotlib.patches import Rectangle, Circle, Arrow
from matplotlib.ticker import NullLocator
from Src.Utils.utils import Space, binaryEncoding
import time

"""
TODO:
IMP: NORMALIZE THE STATE VALUES and decide the max and min

1. Add stochastic noise during motion
3. Assert step size is less than wall width
4. Reward shaping
5. **IMP: Need to give Goal reward on reaching goal and not from taking action from goal 
          but episode should be terminated when choosing an action from goal state
7. Assert seected actions are withing range
"""
# DTYPE = np.float32
# ctypedef np.float32_t DTYPE_t

class Gridworld_cont(object):
    def __init__(self,
                 action_type='continuous',  # 'discrete' {0,1} or 'continuous' [0,1]
                 n_actions=2,
                 debug=True,
                 max_step_length=0.2,
                 max_steps=30):

        self.debug = debug

        self.n_actions = 2
        self.action_type = action_type
        self.action_space = Space(low=-np.ones(self.n_actions)/1.415, high=np.ones(self.n_actions)/1.415, dtype=np.float32) #max range is 1/sqrt(2)
        self.observation_space = Space(low=np.zeros(2, dtype=np.float32), high=np.ones(2, dtype=np.float32), dtype=np.float32)
        self.disp_flag = False

        self.wall_width = 0.05
        self.step_unit = self.wall_width  - 0.005
        self.repeat = int(max_step_length/self.step_unit)

        self.max_steps = int(max_steps/max_step_length)
        self.step_reward = -0.05
        self.collision_reward = 0#-0.05
        self.movement_reward = 0#1
        self.randomness = 0.1

        self.n_lidar = 0
        self.angles = np.linspace(0, 2*np.pi, self.n_lidar+1)[:-1] # Get 10 lidar directions, (11th and 0th are same)
        # self.lidar_angles = np.array(list(zip(np.cos(self.angles), np.sin(self.angles))), dtype=np.float32)
        self.lidar_angles = list(zip(np.cos(self.angles), np.sin(self.angles)))
        self.static_obstacles = self.get_static_obstacles()

        if debug:
            self.heatmap_scale = 99
            self.heatmap = np.zeros((self.heatmap_scale+1, self.heatmap_scale+1))

        self.reset()
        # self.t = 0

    def seed(self, seed):
        self.seed = seed #Unused

    def render(self):
        x, y = self.curr_pos

        # ----------------- One Time Set-up --------------------------------
        if not self.disp_flag:
            self.disp_flag = True
            # plt.axis('off')
            self.currentAxis = plt.gca()
            plt.figure(1, frameon=False)                            #Turns off the the boundary padding
            self.currentAxis.xaxis.set_major_locator(NullLocator()) #Turns of ticks of x axis
            self.currentAxis.yaxis.set_major_locator(NullLocator()) #Turns of ticks of y axis
            plt.ion()                                               #To avoid display blockage

            self.circle = Circle((x, y), 0.01, color='red')
            for coords in self.static_obstacles:
                x1, y1, x2, y2 = coords
                w, h = x2-x1, y2-y1
                self.currentAxis.add_patch(Rectangle((x1, y1), w, h, fill=True, color='gray'))
            print("Init done")
        # ----------------------------------------------------------------------

        for key, val in self.dynamic_obs.items():
            coords, cond = val
            if cond:
                x1, y1, x2, y2 = coords
                w, h = x2 - x1, y2 - y1
                self.objects[key] = Rectangle((x1, y1), w, h, fill=True, color='black')
                self.currentAxis.add_patch(self.objects[key])


        for key, val in self.reward_states.items():
            coords, cond = val
            if cond:
                x1, y1, x2, y2 = coords
                w, h = x2 - x1, y2 - y1
                self.objects[key] = Rectangle((x1, y1), w, h, fill=True)
                self.currentAxis.add_patch(self.objects[key])

        if len(self.angles) > 0:
            r = self.curr_state[-10:]
            coords = zip(r*np.cos(self.angles), r*np.sin(self.angles))

            for i, (w, h) in enumerate(coords):
                self.objects[str(i)] = Arrow(x, y, w, h, width=0.01, fill=True, color='lightgreen')
                self.currentAxis.add_patch(self.objects[str(i)])


        self.objects['circle'] = Circle((x, y), 0.01, color='red')
        self.currentAxis.add_patch(self.objects['circle'])

        # remove all the dynamic objects
        plt.pause(1e-7)
        for _, item in self.objects.items():
            item.remove()
        self.objects = {}

    def set_rewards(self):
        # All rewards
        self.G1_reward = 100 #100
        self.G2_reward = 0

    def reset(self):
        """
        Sets the environment to default conditions
        :return: None
        """
        self.set_rewards()
        self.steps_taken = 0
        self.reward_states = self.get_reward_states()
        self.dynamic_obs = self.get_dynamic_obstacles()
        self.objects = {}

        #x = 0.25
        #x = np.clip(x + np.random.randn()/30, 0.15, 0.35) # Add noise to initial x position
        self.curr_pos = np.array([0.25, 0.1], dtype=np.float64)
        # self.curr_action = np.zeros(self.n_actions, dtype=np.float64)
        # self.curr_state = np.zeros(2+10, dtype=np.float64)
        # self.update_state()

        self.curr_state = self.make_state()
        return self.curr_state


    def step(self, action):
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)  # Clip actions to the valid range
        self.steps_taken += 1
        reward = 0

        # Check if previous state was end of MDP, if it was, then we are in absorbing state currently.
        # Terminal state has a Self-loop and a 0 reward
        term =  self.is_terminal()
        if term:
            return self.curr_state, 0, term, {'No INFO implemented yet'}

        reward += self.step_reward
        for i in range(self.repeat):
            if np.random.rand() < self.randomness:
                #Add noise some percentage of the time
                noise = np.random.rand(2)/1.415 # normalize by max L2 of noise
                delta = noise * self.step_unit # Add noise some percentage of the time
            else:
                delta = clipped_action * self.step_unit

            new_pos = self.curr_pos + delta  # Take a unit step in the direction of chosen action

            if self.valid_pos(new_pos):
                dist = np.linalg.norm(delta)
                reward += self.movement_reward * dist # small reward for moving
                if dist >= self.wall_width:
                    print("ERROR: Step size bigger than wall width", new_pos, self.curr_pos, dist, delta, clipped_action, self.step_unit)

                self.curr_pos = new_pos
                reward += self.get_goal_rewards(self.curr_pos)
                # reward += self.open_gate_condition(self.curr_pos)
            else:
                reward += self.collision_reward
                break

            # To avoid overshooting the goal
            if self.is_terminal():
                break

            # self.update_state()
            self.curr_state = self.make_state()

        if self.debug:
            # Track the positions being explored by the agent
            x_h, y_h = self.curr_pos*self.heatmap_scale
            self.heatmap[min(int(y_h), 99), min(int(x_h), 99)] += 1

            ## For visualizing obstacle crossing flaw, if any
            # for alpha in np.linspace(0,1,10):
            #     mid = alpha*prv_pos + (1-alpha)*self.curr_pos
            #     mid *= self.heatmap_scale
            #     self.heatmap[min(int(mid[1]), 99)+1, min(int(mid[0]), 99)+1] = 1


        return self.curr_state.copy(), reward, self.is_terminal(), {'No INFO implemented yet'}


    def make_state(self):
        x, y = self.curr_pos
        state = [x, y]

        # Append lidar values
        #TODO: Make these loops faster with cython
        for cosine, sine in self.lidar_angles:
            r, r_prv = 0, 0
            pos = (x+r*cosine, y+r*sine)
            while self.valid_pos(pos) and r < 0.5:
                r_prv = r
                r += self.step_unit
                pos = (x+r*cosine, y+r*sine)
            state.append(r_prv)

        # Append the previous action chosen
        # state.extend(self.curr_action)

        return state

    # def update_state(self):
    #     # x, y = self.curr_pos
    #     # state = [x, y]
    #     # Update x, y coordinates
    #     self.curr_state[:2] = self.curr_pos
    #
    #     # Append lidar values
    #     # cdef float cosine_sine
    #     # cdef int idx = 2
    #     idx = 2
    #     # cdef float r = 0
    #     r = 0
    #     for cosine_sine in self.lidar_angles:
    #         r = 0
    #         pos = self.curr_pos + r*cosine_sine
    #         while self.valid_pos(pos) and r < 0.5:
    #             r += self.step_unit
    #             pos = self.curr_pos + r*cosine_sine
    #
    #         self.curr_state[idx] = r
    #         idx += 1
    #
    #     # Append the previous action chosen
    #     # state.extend(self.curr_action)
    #     # return state

    def get_goal_rewards(self, pos):
        for key, val in self.reward_states.items():
            region, reward = val
            if reward and self.in_region(pos, region):
                self.reward_states[key] = (region, 0) #remove reward once taken
                if self.debug: print("Got reward {} in {} steps!! ".format(reward, self.steps_taken))

                return reward
        return 0

    def get_reward_states(self):
        # self.G1 = np.array([0.25, 0.45, 0.30, 0.5], dtype=np.float32)
        # self.G2 = np.array([0.70, 0.85, 0.75, 0.90], dtype=np.float32)

        self.G1 = (0.25, 0.45, 0.30, 0.5)
        self.G2 = (0.70, 0.85, 0.75, 0.90)
        return {'G1': (self.G1, self.G1_reward),
                'G2': (self.G2, self.G2_reward)}

    def get_dynamic_obstacles(self):
        """
        :return: dict of objects, where key = obstacle shape, val = on/off
        """
        return {}

        # self.Gate = (0.15,0.25,0.35,0.3)
        # return {'Gate': (self.Gate, self.Gate_reward)}

    def get_static_obstacles(self):
        """
        Each obstacle is a solid bar, represented by (x,y,x2,y2)
        representing bottom left and top right corners,
        in percentage relative to board size

        :return: list of objects
        """
        # self.O1 = np.array([0,0.25,0.5,0.3], dtype=np.float32)
        # self.O2 = np.array([0.5,0.25,0.55,0.8], dtype=np.float32) #(0.5,0.25,0.54,0.8)
        self.O1 = (0.0, 0.25, 0.50, 0.30)
        # self.O2 = (0.0, 0.0, 0.0, 0.0)
        self.O2 = (0.5, 0.25, 0.55, 0.80)
        obstacles = [self.O1, self.O2]
        return obstacles

    # def valid_pos(self, np.ndarray[DTYPE_t, ndim=1] pos):
    def valid_pos(self, pos):
        flag = True

        # Check boundary conditions
        if not self.in_region(pos, [0,0,1,1]):
            flag = False

        # Check collision with static obstacles
        for region in self.static_obstacles:
            if self.in_region(pos, region):
                flag = False
                break

        # Check collision with dynamic obstacles
        for key, val in self.dynamic_obs.items():
            region, cond = val
            if cond and self.in_region(pos, region):
                flag = False
                break

        return flag

    def is_terminal(self):
        if self.in_region(self.curr_pos, self.G1):
            return 1
        elif self.steps_taken >= self.max_steps:
            return 1
        else:
            return 0

    # def in_region(self, pos, region):
    #     return self.in_region2(np.asarray(pos, dtype=np.float32), np.asarray(region, dtype=np.float32))
    # def in_region(self, float[:] pos, float[:] region):
    # def in_region(self, np.ndarray[DTYPE_t, ndim=1] pos, np.ndarray[DTYPE_t, ndim=1] region):

    # def in_region(self, pos, region):
    #     cdef float x0 = pos[0]
    #     cdef float y0 = pos[1]
    #     cdef float x1 = region[0]
    #     cdef float y1 = region[1]
    #     cdef float x2 = region[2]
    #     cdef float y2 = region[3]
    #     if x0 >= x1 and x0 <= x2 and y0 >= y1 and y0 <= y2:
    #     # if pos[0] >= region[0] and pos[0] <= region[2] and pos[1] >= region[1] and pos[1] <= region[3]:
    #         return True
    #     else:
    #         return False

    def in_region(self, pos, region):
        x0, y0 = pos
        x1, y1, x2, y2 = region
        if x0 >= x1 and x0 <= x2 and y0 >= y1 and y0 <= y2:
            return True
        else:
            return False


if __name__=="__main__":
    # Random Agent
    env = Gridworld_cont(debug=True, n_actions=8)
    for i in range(5):
        done = False
        env.reset()
        while not done:
            env.render()
            action = np.random.randn(env.n_actions)
            next_state, r, done, _ = env.step(action)
