import argparse
from datetime import datetime

"""
Environments:

Gym:

sudo apt install cmake
sudo apt install zlib1g-dev 
pip install gym[atari]

    - BipedalWalker-v2
    - Pendulum-v0
    - MountainCarContinuous-v0
    - MountainCar-v0
    - CartPole-v0
    - ...
    -           
    - Riverraid-ram-v0
    - Alien-ram-v0
    - Seaquest-ram-v0
    - RoadRunner-ram-v0

    
Toy-Domains:
    - Gridworld
    - Gridworld_simple
    - Gridworld_simple2
    - Gridworld_cont
    
Recommender Domains:
    - adobe-helpx
    - adobe-highbeam
    - adobe-moodboard
    
=======================================

Algorithms:
    - ActorCritic 
    - Reinforce 
    - ActorCritic_np
    - DPG
    - NAC
    - TD-lambda
    
    - embed_ActorCritic
    - embed_DPG 
    - embed_Reinforce

"""


class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        # Parameters for Hyper-param sweep
        parser.add_argument("--base", default=-2, help="Base counter for Hyper-param search", type=int)
        parser.add_argument("--inc", default=-2, help="Increment counter for Hyper-param search", type=int)
        parser.add_argument("--hyper", default=-2, help="Which Hyper param settings", type=int)
        parser.add_argument("--seed", default=12345, help="seed for variance testing", type=int)


        # General parameters
        parser.add_argument("--save_count", default=1000, help="Number of ckpts for saving results and model", type=int)
        parser.add_argument("--optim", default='sgd', help="Optimizer type", choices=['adam', 'sgd', 'rmsprop'])
        parser.add_argument("--log_output", default='term_file', help="Log all the print outputs",
                            choices=['term_file', 'term', 'file'])
        parser.add_argument("--debug", default=True, type=self.str2bool, help="Debug mode on/off")
        parser.add_argument("--restore", default=False, type=self.str2bool, help="Retrain flag")
        parser.add_argument("--save_model", default=True, type=self.str2bool, help="flag to save model ckpts")
        parser.add_argument("--summary", default=True, type=self.str2bool, help="--UNUSED-- Visual summary of various stats")
        parser.add_argument("--gpu", default=0, help="GPU BUS ID ", type=int)

        # Book-keeping parameters
        now = datetime.now()
        timestamp = str(now.month) + '|' + str(now.day) + '|' + str(now.hour) + ':' + str(now.minute) + ':' + str(
            now.second)
        parser.add_argument("--timestamp", default=timestamp, help="Timestamp to prefix experiment dumps")
        parser.add_argument("--folder_suffix", default='Default', help="folder name suffix")
        parser.add_argument("--experiment", default='Test_run', help="Name of the experiment")

        self.Env_n_Agent_args(parser)  # Decide the Environment and the Agent
        self.Main_AC_args(parser)  # General Basis, Policy, Critic
        self.PPO_args(parser)  # PPO
        self.DPG_args(parser)  # DPG specific
        self.CL_args(parser)  # Settings for Continual Learning
        self.Action_embed_args(parser)  # Settings for Embedding code
        self.Hyperbolic_embed_args(parser)  # Settings for Hyperbolic Embedding code
        self.SAS_PG(parser)  # Settings for stochastic action set

        self.parser = parser

    def str2bool(self, text):
        if text == 'True':
            arg = True
        elif text == 'False':
            arg = False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        return arg

    def get_parser(self):
        return self.parser


    def Env_n_Agent_args(self, parser):
        # parser.add_argument("--algo_name", default='CL_Vanilla_ActorCritic', help="Learning algorithm")
        # parser.add_argument("--algo_name", default='embed_ActorCritic', help="Learning algorithm")
        parser.add_argument("--algo_name", default='embed_DPG_basis', help="Learning algorithm")
        # parser.add_argument("--algo_name", default='Deep_AC', help="Learning algorithm")
        # parser.add_argument("--algo_name", default='discrete_Macro', help="Learning algorithm")
        # parser.add_argument("--algo_name", default='ActorCritic', help="Learning algorithm")
        # parser.add_argument("--algo_name", default='CL_ActorCritic', help="Learning algorithm")
        # parser.add_argument("--env_name", default='Seaquest-ram-v0', help="Environment to run the code")
        # parser.add_argument("--env_name", default='MountainCarContinuous-v0', help="Environment to run the code")
        parser.add_argument("--env_name", default='helpx', help="Environment to run the code")
        # parser.add_argument("--env_name", default='Gridworld_vanilla', help="Environment to run the code")
        parser.add_argument("--n_actions", default=12, help="number of base actions for gridworld", type=int)
        parser.add_argument("--difficulty", default=1, help="Difficulty for six room task", type=int)

        parser.add_argument("--max_episodes", default=int(1e1), help="maximum number of episodes (75000)", type=int)
        parser.add_argument("--max_steps", default=150, help="maximum steps per episode (500)", type=int)


    def PPO_args(self, parser):
        parser.add_argument("--IS_clip", default=0.1, help="Permissable deviation for importance sampling", type=float)
        parser.add_argument("--n_steps", default=8, help="n-step returns", type=int)
        parser.add_argument("--inner_epoch", default=4, help="Inner epoch over the n-step data", type=int)

    def DPG_args(self, parser):
        parser.add_argument("--tau", default=0.001, help="soft update regularizer", type=float)

    def CL_args(self, parser):
        parser.add_argument("--re_init", default='none', help="(none, policy, full) Reinitialize parameters on change")
        parser.add_argument("--freeze_action_rep", default=False, help="Freeze prv action rep on change", type=self.str2bool)  #UNUSED
        parser.add_argument("--change_count", default=1, help="Number of CL changes", type=float)
        parser.add_argument("--dynamic_reward_action", default=False, help="Do dynamic actions have associated rewards?", type=self.str2bool)
        parser.add_argument("--uniform_action_spread", default=True, help="Actions are uniformly spread", type=self.str2bool)

    def Action_embed_args(self, parser):
        parser.add_argument("--valid_fraction", default=0.2, help="Fraction of data used for validation", type=float)
        parser.add_argument("--true_embeddings", default=False, help="Use ground truth embeddings or not?", type=self.str2bool)
        parser.add_argument("--only_phase_one", default=False, help="Only phase1 training", type=self.str2bool)
        parser.add_argument("--emb_lambda", default=0, help="Lagrangian for learning embedding on the fly", type=float)
        parser.add_argument("--embed_lr", default=1e-4, help="Learning rate of action embeddings", type=float)
        parser.add_argument("--emb_reg", default=1e-2, help="L2 regularization for embeddings", type=float)
        parser.add_argument("--beta_vae", default=1e-2, help="Lagrangian for KL penalty", type=float)
        parser.add_argument("--emb_fraction", default=1, help="--UNUSED-- fraction of embeddings to consider", type=float)
        parser.add_argument("--reduced_action_dim", default=32, help="dimensions of action embeddings", type=int)
        parser.add_argument("--load_embed", default=False, type=self.str2bool, help="Retrain flag")

        parser.add_argument("--sup_batch_size", default=64, help="(64)Supervised learning Batch size", type=int)
        parser.add_argument("--initial_phase_epochs", default=1000, help="maximum number of episodes (150)", type=int)

    def Hyperbolic_embed_args(self, parser):
        parser.add_argument("--macro_diversity", default='bfs', help="Max length of any macro",  choices=['bfs', 'dfs', 'primitive'])
        parser.add_argument("--max_macro_length", default=4, help="Max length of any macro IN BFS", type=int)
        parser.add_argument("--expected_macro_length", default=5, help="Expected length of a macro", type=int)
        parser.add_argument("--max_num_macro", default=500, help="Max numbers of macro", type=int)
        parser.add_argument("--prior_reg", default=10, help="Regularizer for hierarchy prior", type=float)
        parser.add_argument("--rnn_type", default='RNN', help="rnn for macro embeddings", choices=['RNN', 'GRU', 'LSTM'])
        parser.add_argument("--rnn_hidden_dim", default=32, help="Size of hidden dimension in rnn", type=int)
        parser.add_argument("--rnn_hidden_layers", default=1, help="Number of hidden layers in rnn", type=int)
        parser.add_argument("--rnn_dropout", default=0, help="Dropouts for the hidden layers in rnn", type=float)
        parser.add_argument("--poincare_eps", default=1e-5, help="Boundary for poincare space", type=float)
        parser.add_argument("--train_embedding", default='RL', help="train macro embeddings on the fly (None, RL, sup)")

    def SAS_PG(self, parser):
        parser.add_argument("--q_lr", default=1e-2, help="Learning rate for Q", type=float)
        parser.add_argument("--v_lr", default=1e-3, help="Learning rate for V", type=float)
        parser.add_argument("--alpha_rate", default=0.999, help="Mixing momentum", type=float)
        parser.add_argument("--action_prob", default=0.8, help="Action available probability", type=float)
        parser.add_argument("--SAS_q_updates", default=8, help="Number of batches per optim step", type=int)

    def Main_AC_args(self, parser):
        parser.add_argument("--exp", default=0.999, help="Eps-greedy epxloration decay", type=float)
        parser.add_argument("--gamma", default=0.999, help="Discounting factor", type=float)
        parser.add_argument("--trace_lambda", default=0.9, help="Lambda returns", type=float)
        parser.add_argument("--actor_lr", default=1e-3, help="Learning rate of actor", type=float)
        parser.add_argument("--critic_lr", default=1e-2, help="Learning rate of critic/baseline", type=float)
        parser.add_argument("--state_lr", default=1e-3, help="Learning rate of state features", type=float)
        parser.add_argument("--gauss_variance", default=0.25, help="Variance for gaussian policy", type=float)
        parser.add_argument("--entropy_lambda", default=0.01, help="Lagrangian for policy's entropy", type=float)

        parser.add_argument("--Natural", default=False, help="--UNUSED--Natural gradient", type=self.str2bool)
        parser.add_argument("--fourier_coupled", default=True, help="Coupled or uncoupled fourier basis", type=self.str2bool)
        parser.add_argument("--fourier_order", default=-1, help="Order of fourier basis, " +
                                                               "(if > 0, it overrides neural nets)", type=int)
        parser.add_argument("--NN_basis_dim", default='256', help="Shared Dimensions for Neural network layers")
        parser.add_argument("--Policy_basis_dim", default='2,16', help="Dimensions for Neural network layers for policy")

        parser.add_argument("--buffer_size", default=int(1e5), help="Size of memory buffer (3e5)", type=int)
        parser.add_argument("--batch_size", default=1, help="Batch size", type=int)
