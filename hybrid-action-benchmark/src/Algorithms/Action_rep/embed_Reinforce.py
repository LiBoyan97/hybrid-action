import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from Src.Utils.utils import NeuralNet, MemoryBuffer
from Src.Algorithms.Agent import Agent
from Src.Utils import Basis, Policy, Critic

"""
TODO:
"""

class Action_representation(NeuralNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 config):
        super(Action_representation, self).__init__()

        #TODO: define embeddings as nn.Embeddings and use sparse gradients
        #TODO: Look into hierarchical/adaptive softmax for embeddings
        # https://github.com/rosinality/adaptive-softmax-pytorch
        # https://gist.github.com/paduvi/588bc95c13e73c1e5110d4308e6291ab
        # http://ruder.io/word-embeddings-softmax/

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # Action embeddings to project the predicted action into original dimensions
        if config.true_embeddings:
            embeddings = config.env.motions.copy()
            embeddings = np.expand_dims(embeddings, axis=0)
            maxi, mini = np.max(embeddings), np.min(embeddings)
            embeddings = ((embeddings - mini)/(maxi-mini))*2 - 1  # Normalize to (-1, 1)
            self.embeddings = Variable(torch.from_numpy(embeddings).type(config.dtype), requires_grad=False)
            self.reduced_action_dim = np.shape(embeddings)[2]
        else:
            self.reduced_action_dim = config.reduced_action_dim
            uniform_tensor = torch.rand(1, self.action_dim, self.reduced_action_dim)*4 - 2  # Tanh of this range will be mostly around (-1, 1)
            self.embeddings = torch.nn.Parameter(uniform_tensor.type(config.dtype), requires_grad=True)

        # One layer neural net to get action representation
        self.fc1 = nn.Linear(self.state_dim*2, self.reduced_action_dim)
        # self.fc2 = nn.Linear(128, self.reduced_action_dim)

        print("Action representation: ", [m for m, _ in self.named_parameters()])
        self.optim = config.optim(self.parameters(), lr=self.config.embed_lr)
        # self.optim = torch.optim.Adam(self.parameters(), lr=self.config.embed_lr)

    def get_match_scores(self, action):
        # compute similarity probability based on L2 norm

        # Action \in Batch * dim and Embedding \in 1 * Actions * dim
        # we want result \in Batch * Actions, therefore unsqueeze to broadcast accordingly
        action = action.unsqueeze(1)
        embeddings = self.embeddings
        if not self.config.true_embeddings:
            embeddings = F.tanh(self.embeddings)
        diff = torch.norm(embeddings - action, p=2, dim=-1)
        return diff


    def get_best_match(self, action):
        diff = self.get_match_scores(action)
        val, pos = torch.min(diff, dim=1)
        # print("-----", action, diff, pos)
        return pos.cpu().data.numpy()[0]


    def get_match_dist(self, action):
        diff = self.get_match_scores(action)
        probs = F.softmax(-diff, dim=-1)       # probs = F.softmax(1.0/(diff+1e-10), dim=-1)

        ## Dot product based similarity
        # if not self.config.true_embeddings:
        #     embeddings = F.tanh(self.embeddings)
        # else:
        #     embeddings = self.embeddings
        #
        # sim = torch.mm(action, embeddings)
        # probs = F.softmax(sim, dim=-1)
        return probs

    def forward(self, state1, state2):
        # concatenate the state features and predict the action required to go from state1 to state2
        state_cat = torch.cat([state1, state2], dim=1)
        x = F.tanh(self.fc1(state_cat))
        # x = F.tanh(self.fc2(x))
        return self.get_match_dist(x)

    def get_embedding(self, action):
        # Get the corresponding target embedding
        action_emb = self.embeddings[:, action]
        if not self.config.true_embeddings:
            action_emb = F.tanh(action_emb)
        return action_emb

class embed_Reinforce(Agent):
    def __init__(self, config):
        super(embed_Reinforce, self).__init__(config)

        self.ep_rewards = []
        self.ep_states = []
        self.ep_actions = []
        self.ep_exec_action_embs = []
        self.ep_chosen_action_embs = []

        # Set Hyper-parameters
        self.memory = MemoryBuffer(size=config.buffer_size)
        self.counter = 0

        self.initial_phase = not config.true_embeddings  # Initial training phase required if learning embeddings

        # Function to get state features and action representation
        if config.fourier_order > 0:
            self.state_features = Basis.Fourier_Basis(config=config)
        else:
            self.state_features = Basis.NN_Basis(config=config)

        # Function to get state features and action representation
        self.action_rep = Action_representation(state_dim=self.state_features.feature_dim, action_dim=self.action_dim,
                                                config=config)
        self.baseline = Critic.Critic(state_dim=self.state_features.feature_dim, config=config)

        # Create instances for Actor and Q_fn
        self.atype = config.dtype
        self.actor = Policy.embed_Gaussian(action_dim=self.action_rep.reduced_action_dim,
                           state_dim=self.state_features.feature_dim, config=config)
        self.action_size = self.action_dim


        self.modules = [('actor', self.actor), ('baseline', self.baseline),
                        ('state_features', self.state_features), ('action_rep', self.action_rep)]

        self.init()

    def get_action(self, state, explore=0.2):
        explore = 0  #Don't do eps-greedy with policy gradients.
        if self.initial_phase or np.random.rand() < explore:
            # take random actions (uniformly in actual action space) to observe the interactions initially
            action = np.random.randint(self.action_dim)
            exec_action_emb = self.action_rep.get_embedding(action).cpu().view(-1).data.numpy()
            chosen_action_emb = exec_action_emb
        else:
            state = np.float32(state)
            if len(state.shape) == 1:
                state = np.expand_dims(state, 0)

            state = self.state_features.forward(state)
            chosen_action_emb = self.actor.get_action_wo_dist(state, explore=0)
            action = self.action_rep.get_best_match(chosen_action_emb)

            exec_action_emb = self.action_rep.get_embedding(action).cpu().view(-1).data.numpy()
            chosen_action_emb = chosen_action_emb.cpu().view(-1).data.numpy()

        return action, (exec_action_emb, chosen_action_emb)

    def update(self, s1, a1, a_emb1, r1, s2, done):
        if not self.initial_phase:
            # Store the episode history
            self.ep_rewards.append(r1)
            self.ep_states.append(s1)
            self.ep_actions.append(int(a1))
            self.ep_exec_action_embs.append(a_emb1[0])
            self.ep_chosen_action_embs.append(a_emb1[1])
            if done:
                # Compute gamma return and do on-policy update
                g_rewards, R = [], 0
                for r in self.ep_rewards[::-1]:
                    R = r + self.config.gamma * R
                    g_rewards.insert(0, R)
                self.optimize(np.float32(self.ep_states), np.float32(self.ep_actions),
                              np.float32(self.ep_exec_action_embs), np.float32(self.ep_chosen_action_embs),
                              np.float32(g_rewards))

                # Reset the episode history
                self.ep_rewards = []
                self.ep_states = []
                self.ep_actions = []
                self.ep_exec_action_embs = []
                self.ep_chosen_action_embs = []

        else:
            self.memory.add(s1, a1, a_emb1[0], r1, s2, int(done != 1), randomize=True)  # a_emb1 gets ignored subsequently
            if self.memory.length >= self.config.buffer_size:
                # action embeddings can be learnt offline
                self.initial_phase_training(max_epochs=self.config.initial_phase_epochs)


    def optimize(self, s1, a1, exec_a1_emb, chosen_a1_emb, r1):
        r1 = Variable(torch.from_numpy(r1).type(self.config.dtype), requires_grad=False).view(-1, 1)
        exec_a1_emb = Variable(torch.from_numpy(exec_a1_emb).type(self.config.dtype), requires_grad=False)
        chosen_a1_emb = Variable(torch.from_numpy(chosen_a1_emb).type(self.config.dtype), requires_grad=False)

        a1_emb = exec_a1_emb if self.config.emb_flag == 'exec' else chosen_a1_emb

        s1 = self.state_features.forward(s1)

        # ---------------------- optimize critic ----------------------
        val_pred = self.baseline.forward(s1)
        # loss_baseline = F.smooth_l1_loss(val_pred, r1)
        loss_baseline = F.mse_loss(val_pred, r1)

        # ---------------------- optimize actor ----------------------
        td_error = (r1 - val_pred).detach()

        if self.config.TIS:
            _, dist = self.actor.get_action(s1)
            exec_prob = self.actor.get_prob_from_dist(dist, exec_a1_emb, scalar=self.config.TIS_scalar)
            chosen_prob = self.actor.get_prob_from_dist(dist, chosen_a1_emb, scalar=self.config.TIS_scalar)
            TIS_ratio = (exec_prob/chosen_prob).detach()  #TODO: clip this ratio?
            loss_actor = -1.0 * torch.mean(TIS_ratio * td_error * self.actor.get_log_prob_dist(dist, exec_a1_emb))
        else:
            loss_actor = -1.0 * torch.mean(td_error * self.actor.get_log_prob(s1, a1_emb))

        # loss_actor = -1.0 * torch.sum(td_error * self.actor.get_log_prob(s1, a1_emb))
        # loss_actor = -1.0 * torch.mean(torch.mean(r1 * self.actor.get_log_prob(s1, a1_emb), -1)) # without baseline
        loss = loss_baseline + loss_actor
        # print(val_pred, a1_emb)

        # ------------ optimize the embeddings always ----------------
        if not self.config.true_embeddings:
            a1 = Variable(torch.from_numpy(a1).type(self.config.dtype_long), requires_grad=False)
            action_pred = self.action_rep.forward(s1[:-1], s1[1:])
            loss_act_rep = F.cross_entropy(action_pred, a1[:-1])
            loss += loss_act_rep * self.config.emb_lambda

        self.step(loss, clip_norm=10)

    def initial_phase_training(self, max_epochs=-1):
        # change optimizer to Adam for supervised learning
        self.action_rep.optim = torch.optim.Adam(self.action_rep.parameters(), lr=1e-3)
        self.state_features.optim = torch.optim.Adam(self.state_features.parameters(), lr=1e-3)
        initial_losses = []

        print("Inital training phase started...")
        #TODO: Split into train and validation to avoid overfitting
        for counter in range(max_epochs):
            losses = []
            for s1, a1, _, _, s2, _ in self.memory.get_batch(size=self.config.sup_batch_size, randomize=True):
                a1 = Variable(torch.from_numpy(a1).type(self.config.dtype_long), requires_grad=False)

                self.clear_gradients()  # clear all the gradients from last run

                s1 = self.state_features.forward(s1)
                s2 = self.state_features.forward(s2)

                # ------------ optimize the embeddings ----------------
                action_pred = self.action_rep.forward(s1, s2)
                loss_act_rep = F.cross_entropy(action_pred, a1)

                loss_act_rep.backward()
                self.action_rep.optim.step()
                self.state_features.optim.step()

                losses.append(loss_act_rep.cpu().view(-1).data.numpy()[0])

            # print(np.mean(loss))
            initial_losses.append(np.mean(losses))
            if counter % 1 == 0:
                print("Epoch {} loss:: {}".format(counter, np.mean(initial_losses[-10:])))
                #self.save()

            # Terminate initial phase once action representations have converged.
            if len(initial_losses) >= 20 and np.mean(initial_losses[-10:]) >= np.mean(initial_losses[-20:]):
                print("Converged...")
                break

        # Reset the optim to whatever is there in config
        self.action_rep.optim = self.config.optim(self.action_rep.parameters(), lr=self.config.embed_lr)
        self.state_features.optim = self.config.optim(self.state_features.parameters(), lr=self.config.state_lr)

        print('... Initial training phase terminated!')
        self.initial_phase = False
