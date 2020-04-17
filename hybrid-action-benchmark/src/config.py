import sys
from yaml import dump
from os import path
import Src.Utils.utils as utils
import numpy as np
import torch
from collections import OrderedDict

class Config(object):
    def __init__(self, args):

        # SET UP PATHS
        self.paths = OrderedDict()
        self.paths['root'] = path.abspath(path.join(path.dirname(__file__), '..'))

        # Do Hyper-parameter sweep, if needed
        self.idx = args.base + args.inc
        if self.idx >= 0 and args.hyper >= 0:
            # self.hyperparam_sweep = utils.dynamic_load(self.paths['root'], "hyperparam_sweep" + str(args.hyper), load_class=False)
            self.hyperparam_sweep = utils.dynamic_load(self.paths['root'], "random_search" + str(args.hyper), load_class=False)
            # self.hyperparam_sweep = utils.dynamic_load(self.paths['root'], "selected_search" + str(args.hyper), load_class=False)
            self.hyperparam_sweep.set(args, self.idx)
            del self.hyperparam_sweep  # *IMP: CanNOT deepcopy an object having reference to an imported library (DPG)

        # Make results reproducible
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Copy all the variables from args to config
        self.__dict__.update(vars(args))

        # Frequency of saving results and models.
        self.save_after = args.max_episodes // args.save_count if args.max_episodes > args.save_count else args.max_episodes

        # add path to models
        folder_suffix = args.experiment + args.folder_suffix
        self.paths['Experiments'] = path.join(self.paths['root'], 'Experiments')
        self.paths['experiment'] = path.join(self.paths['Experiments'], args.env_name, args.algo_name, folder_suffix)
        path_prefix = [self.paths['experiment'], str(args.seed)]
        self.paths['logs'] = path.join(*path_prefix, 'Logs/')
        self.paths['ckpt'] = path.join(*path_prefix, 'Checkpoints/')
        self.paths['results'] = path.join(*path_prefix, 'Results/')

        # Create directories
        for (key, val) in self.paths.items():
            if key not in ['root', 'datasets', 'data']:
                utils.create_directory_tree(val)

        # Save the all the configuration settings

        open(path.join(self.paths['experiment'], 'args.yaml'), 'w')
        print(path.join(self.paths['experiment'], 'args.yaml'))
        dump(args.__dict__, open(path.join(self.paths['experiment'], 'args.yaml'), 'w'), default_flow_style=False,
             explicit_start=True)

        # Output logging
        sys.stdout = utils.Logger(self.paths['logs'], args.restore, args.log_output)

        # Get the domain and algorithm
        self.env, self.gym_env, self.cont_actions = self.get_domain(args.env_name, args=args, debug=args.debug,
                                                               path=path.join(self.paths['root'], 'Environments'))
        self.env.seed(seed)

        # Get the embedding paths
        self.get_embedding_paths(args)

        # Set Model
        self.algo = utils.dynamic_load(path.join(self.paths['root'], 'Src', 'Algorithms'), args.algo_name, load_class=True)

        self.feature_dim = [int(size) for size in args.NN_basis_dim.split(',')]
        self.policy_basis_dim = [int(size) for size in args.Policy_basis_dim.split(',')]

        # GPU
        self.device = torch.device('cuda' if args.gpu else 'cpu')

        # optimizer
        if args.optim == 'adam':
            self.optim = torch.optim.Adam
        elif args.optim == 'rmsprop':
            self.optim = torch.optim.RMSprop
        elif args.optim == 'sgd':
            self.optim = torch.optim.SGD
        else:
            raise ValueError('Undefined type of optmizer')


        print("=====Configurations=====\n", args)


    def get_embedding_paths(self, args):
        if hasattr(args, 'true_embeddings'):
            if self.env_name == 'Gridworld_CL':
                prefix = 'CL_' if self.fourier_order < 1 else 'CL_Fourier_'
                self.paths['embedding'] = utils.search(self.paths['root'],
                                                       prefix + str(args.change_count) + '_Grid' + str(args.n_actions),
                                                       exact=True)  # +'.pt'
            if self.env_name == 'Gridworld':
                prefix = '' if self.fourier_order < 1 else 'Fourier_'
                self.paths['embedding'] = utils.search(self.paths['root'], prefix + 'Grid' + str(args.n_actions),
                                                       exact=True)  # +'.pt'
            if 'helpx' in self.env_name:
                self.paths['embedding'] = utils.search(self.paths['root'], 'helpx_8', exact=True)  # +'.pt'

            if self.env_name == 'six_room':
                prefix = ''
                if args.algo_name == 'euc_Macro':
                    prefix = 'tp_euc_'
                elif args.algo_name == 'hyp_Macro':
                    prefix = 'tp_hyp_'

                self.paths['embedding'] = utils.search(self.paths['root'], prefix + 'emb_' + str(args.reduced_action_dim)
                                                       + '_' + str(args.max_num_macro), exact=True)
                self.paths['macro_list'] = utils.search(self.paths['root'], prefix + 'macro_' + str(args.reduced_action_dim)
                                                        + '_' + str(args.max_num_macro) + '.npy', exact=True)

            if (not self.gym_env) and self.true_embeddings:
                self.reduced_action_dim = self.env.get_embeddings().shape[1]


    def get_domain(self, tag, args, path, debug=True):
        if tag == 'helpx' or tag == 'highbeam':
            obj = utils.dynamic_load(path, 'RecommenderMDP', load_class=True)
            env = obj(file=tag, ngram_n=2, debug=debug)
            return env, False, False

        elif tag == 'helpx_CL' or tag == 'highbeam_CL':
            obj = utils.dynamic_load(path, 'RecommenderMDP_CL', load_class=True)
            env = obj(file=tag[:-3], ngram_n=2, debug=debug, change_count=self.change_count,
                      dynamic_reward_action=self.dynamic_reward_action, max_episodes=self.max_episodes, change_remove=args.change_remove)
            return env, False, False

        elif tag == 'Gridworld_CL':
            obj = utils.dynamic_load(path, tag, load_class=True)
            env = obj(n_actions=args.n_actions, uniform_action_spread=self.uniform_action_spread,
                      change_count=self.change_count, debug=debug, max_episodes=self.max_episodes, change_remove=args.change_remove)
            return env, False, env.action_space.dtype == np.float32

        elif tag == 'Gridworld' or tag == 'Gridworld_vanilla':
            obj = utils.dynamic_load(path, tag, load_class=True)
            env = obj(n_actions=args.n_actions, debug=debug)
            return env, False, env.action_space.dtype == np.float32

        elif tag == 'six_room':
            obj = utils.dynamic_load(path, tag, load_class=True)
            env = obj(difficulty=args.difficulty, debug=debug)
            return env, False, env.action_space.dtype == np.float32

        elif tag == 'Gridworld_SAS' or tag == 'SAS_Bandit' or tag == 'SAS_Reco':
            obj = utils.dynamic_load(path, tag, load_class=True)
            env = obj(n_actions=args.n_actions, debug=debug, action_prob=args.action_prob)
            return env, False, env.action_space.dtype == np.float32

        elif tag == 'SF_Map':
            obj = utils.dynamic_load(path, tag, load_class=True)
            env = obj(debug=debug, action_prob=args.action_prob, bridge_prob=args.SF_bridge_prob)
            return env, False, env.action_space.dtype == np.float32

        else:
            try:
                import gym
                from gym.spaces.box import Box
                env = gym.make(tag)
                return env, True, isinstance(env.action_space, Box)
            except:
                raise ValueError("Error! Environment neither available in Gym nor locally.")

if __name__ == '__main__':
    pass