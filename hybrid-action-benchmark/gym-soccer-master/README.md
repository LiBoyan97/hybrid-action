# gym-soccer

The [Soccer environment](https://github.com/LARG/HFO) is a multiagent
domain featuring continuous state and action spaces. 

## Changes

Several changes have been made to more closely reflect the setup used by [[Hausknecht & Stone 2016]](https://arxiv.org/abs/1511.04143):

- The number of steps without touching the ball before ending an episode has been reduced to 100.
- The reward function has abeen updated to reflect the one used in their code (https://github.com/mhauskn/dqn-hfo). Specifically, the negative reward given for the distance between the ball and goal is only activated once the agent is in possession of the ball. A separate environment has been created with this change: `SoccerScoreGoal-v0`. It is the same as `SoccerEmptyGoal-v0` except for the reward function.
- The state of the environment is returned after each step (useful for counting the number of goals).
## Tasks

There are several tasks supported at the moment:

### Soccer

The soccer task initializes a single offensive agent on the field and rewards +1 for scoring a goal and 0 otherwise. In order to score a goal, the agent will need to know how to approach the ball and kick towards the goal. The sparse nature of the goal reward makes this task very difficult to accomplish.

### SoccerEmptyGoal

The SoccerEmptyGoal task features a more informative reward signal than the Soccer task. As before, the objective is to score a goal. However, SoccerEmtpyGoal rewards the agent for approaching the ball and moving the ball towards the goal. These frequent rewards make the task much more accessible.

### SoccerAgainstKeeper

The objective of the SoccerAgainstKeeper task is to score against a goal keeper. The agent is rewarded for moving the ball towards the goal and for scoring a goal. The goal keeper uses a hand-coded policy developed by the Helios RoboCup team. The difficulty in this task is learning how to shoot around the goal keeper.


# Installation

```bash
cd gym-soccer
pip install -e .
```

or

```bash
pip install -e git+https://github.com/cycraig/gym-soccer#egg=gym_soccer
```