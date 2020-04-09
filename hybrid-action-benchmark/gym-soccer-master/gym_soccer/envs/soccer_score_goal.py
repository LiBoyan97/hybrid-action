import logging
import math
import numpy as np
from gym import spaces
from gym_soccer.envs.soccer_env import SoccerEnv, ACTION_LOOKUP
from gym_soccer.envs.soccer_empty_goal import SoccerEmptyGoalEnv

try:
    import hfo_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install HFO dependencies with 'pip install gym[soccer].)'".format(e))

logger = logging.getLogger(__name__)

class SoccerScoreGoalEnv(SoccerEmptyGoalEnv):
    """
    SoccerScoreGoal is the same task as SoccerEmptyGoal, which tasks the 
    agent with approaching the ball, dribbling, and scoring a goal. Rewards 
    are given as the agent nears the ball, kicks the ball towards the goal, 
    and scores a goal.

    The difference is that the reward structure is altered to be consistent
    with the Hausknecht paper: "Deep Reinforcement Learning with Parameterised
    Action Spaces".

    """
    def __init__(self):
        super(SoccerScoreGoalEnv, self).__init__()
        # dash, turn, kick, tackle
        low0 = np.array([0, -180], dtype=np.float32)  # meant to be 0, not -100! (according to original soccer env and dqn-hfo inverting gradients)
        high0 = np.array([100, 180], dtype=np.float32)
        low1 = np.array([-180], dtype=np.float32)
        high1 = np.array([180], dtype=np.float32)
        low2 = np.array([0, -180], dtype=np.float32)
        high2 = np.array([100, 180], dtype=np.float32)
        low3 = np.array([-180], dtype=np.float32)
        high3 = np.array([180], dtype=np.float32)
        self.action_space = spaces.Tuple((spaces.Discrete(3),
                                          spaces.Box(low=low0, high=high0, dtype=np.float32),
                                          spaces.Box(low=low1, high=high1, dtype=np.float32),
                                          spaces.Box(low=low2, high=high2, dtype=np.float32)))#,
                                          #spaces.Box(low=low3, high=high3)))
                                          
        self.unum = self.env.getUnum()  # uniform number (identifier) of our lone agent
        print("UNUM =",self.unum)
        
    '''def _take_action(self, action):
        """ Converts the action space into an HFO action. """
        action_type = ACTION_LOOKUP[action[0]]
        if action_type == hfo_py.DASH:
            self.env.act(action_type, action[1], action[2])
        elif action_type == hfo_py.TURN:
            self.env.act(action_type, action[3])
        elif action_type == hfo_py.KICK:
            self.env.act(action_type, action[4], action[5])
        elif action_type == hfo_py.TACKLE:
            self.env.act(action_type, action[6])
        else:
            print('Unrecognized action %d' % action_type)
            self.env.act(hfo_py.NOOP)'''
            
    def _get_reward(self):
        """
        Agent is rewarded for minimizing the distance between itself and
        the ball, minimizing the distance between the ball and the goal,
        and scoring a goal.
        """
        current_state = self.env.getState()
        #print("State =",current_state)
        #print("len State =",len(current_state))
        ball_proximity = current_state[53]
        goal_proximity = current_state[15]
        ball_dist = 1.0 - ball_proximity
        goal_dist = 1.0 - goal_proximity
        kickable = current_state[12]
        ball_ang_sin_rad = current_state[51]
        ball_ang_cos_rad = current_state[52]
        ball_ang_rad = math.acos(ball_ang_cos_rad)
        if ball_ang_sin_rad < 0:
            ball_ang_rad *= -1.
        goal_ang_sin_rad = current_state[13]
        goal_ang_cos_rad = current_state[14]
        goal_ang_rad = math.acos(goal_ang_cos_rad)
        if goal_ang_sin_rad < 0:
            goal_ang_rad *= -1.
        alpha = max(ball_ang_rad, goal_ang_rad) - min(ball_ang_rad, goal_ang_rad)
        ball_dist_goal = math.sqrt(ball_dist*ball_dist + goal_dist*goal_dist -
                                   2.*ball_dist*goal_dist*math.cos(alpha))
        # Compute the difference in ball proximity from the last step
        if not self.first_step:
            ball_prox_delta = ball_proximity - self.old_ball_prox
            kickable_delta = kickable - self.old_kickable
            ball_dist_goal_delta = ball_dist_goal - self.old_ball_dist_goal
        self.old_ball_prox = ball_proximity
        self.old_kickable = kickable
        self.old_ball_dist_goal = ball_dist_goal
        #print(self.env.playerOnBall())
        #print(self.env.playerOnBall().unum)
        #print(self.env.getUnum())
        reward = 0
        if not self.first_step:
            '''# Reward the agent for moving towards the ball
            reward += ball_prox_delta
            if kickable_delta > 0 and not self.got_kickable_reward:
                reward += 1.
                self.got_kickable_reward = True
            # Reward the agent for kicking towards the goal
            reward += 0.6 * -ball_dist_goal_delta
            # Reward the agent for scoring
            if self.status == hfo_py.GOAL:
                reward += 5.0'''
            '''reward = self.__move_to_ball_reward(kickable_delta, ball_prox_delta) + \
                    3. * self.__kick_to_goal_reward(ball_dist_goal_delta) + \
                    self.__EOT_reward();'''
            mtb = self.__move_to_ball_reward(kickable_delta, ball_prox_delta)
            ktg = 3. * self.__kick_to_goal_reward(ball_dist_goal_delta)
            eot = self.__EOT_reward()
            reward = mtb + ktg + eot
            #print("mtb: %.06f ktg: %.06f eot: %.06f"%(mtb,ktg,eot))
            
        self.first_step = False
        #print("r =",reward)
        return reward
        
    def __move_to_ball_reward(self, kickable_delta, ball_prox_delta):
        reward = 0.
        if self.env.playerOnBall().unum < 0 or self.env.playerOnBall().unum == self.unum:
            reward += ball_prox_delta;
        if kickable_delta >= 1 and not self.got_kickable_reward:
            reward += 1.
            self.got_kickable_reward = True
        return reward;
        
    def __kick_to_goal_reward(self, ball_dist_goal_delta):
        if(self.env.playerOnBall().unum == self.unum):
            return -ball_dist_goal_delta
        elif self.got_kickable_reward == True:
            return 0.2 * -ball_dist_goal_delta
        return 0.
        
    def __EOT_reward(self):
        if self.status == hfo_py.GOAL:
            return 5.
        #elif self.status == hfo_py.CAPTURED_BY_DEFENSE:
        #    return -1.
        return 0.
    
