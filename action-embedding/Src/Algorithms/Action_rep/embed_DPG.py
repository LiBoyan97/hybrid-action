import numpy as np
import torch
from torch.autograd import Variable
from torch import tensor, float32, long
import torch.nn as nn
import torch.nn.functional as F
from Src.Utils.utils import MemoryBuffer, OrnsteinUhlenbeckActionNoise, soft_update, hard_update
from Src.Algorithms.Agent import Agent
from Src.Utils import Basis, ActionRepresentation

"""
TODO:

1. Q function update should be at the executed embedding rather than the sampled one.
2. Policy should be vice versa, i.e update at sampled instead of executed one.
"""

# class Actor(Basis.Fourier_Basis):
class Actor(Basis.NN_Basis):
    def __init__(self, action_dim, config):
        super(Actor, self).__init__(config=config)

        # Initialize network architecture and optimizer
        self.fc1 = nn.Linear(self.feature_dim, action_dim)
        self.custom_weight_init()
        print("Actor: ", [(m, param.shape) for m, param in self.named_parameters()])
        self.optim = config.optim(self.parameters(), lr=config.actor_lr)

    def get_action(self, state):
        # Output the action embedding
        state = self.forward(state)
        action = F.tanh(self.fc1(state))
        return action

# class Q_fn(Basis.Fourier_Basis):
class Q_fn(Basis.NN_Basis):
    def __init__(self, action_dim, config):
        super(Q_fn, self).__init__(config=config)

        self.fc1 = nn.Linear(action_dim + self.feature_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.custom_weight_init()
        print("Critic: ", [(m, param.shape) for m, param in self.named_parameters()])
        self.optim = config.optim(self.parameters(), lr=config.critic_lr)

    def forward(self, state, action):
        state = super(Q_fn, self).forward(state)
        x = torch.cat((state, action), dim=1)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class embed_DPG(Agent):
    def __init__(self, config):
        super(embed_DPG, self).__init__(config)

        # Set Hyper-parameters

        self.initial_phase =False# not config.true_embeddings and not config.load_embed  # Initial training phase required if learning embeddings
        self.batch_norm = False
        self.ctr = 0

        # Function to get state features and action representation
        self.action_rep = ActionRepresentation.Action_representation_deep(action_dim=self.action_dim, config=config)
        # Create instances for Actor and Q_fn
        self.actor = Actor(action_dim=self.action_rep.reduced_action_dim, config=config)
        self.Q = Q_fn(action_dim=self.action_rep.reduced_action_dim, config=config)

        # Create target networks
        # Deepcopy not working.
        self.target_actor = Actor(action_dim=self.action_rep.reduced_action_dim, config=config)
        self.target_Q = Q_fn(action_dim=self.action_rep.reduced_action_dim, config=config)
        # self.target_action_rep = ActionRepresentation.Action_representation_deep(action_dim=self.action_dim, config=config)
        # Copy the initialized values to target
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_Q.load_state_dict(self.Q.state_dict())
        # self.target_action_rep.load_state_dict(self.action_rep.state_dict())



        self.memory = MemoryBuffer(max_len=self.config.buffer_size, state_dim=self.state_dim,
                                     action_dim=1, atype=long, config=config,
                                     dist_dim=self.action_rep.reduced_action_dim)  # off-policy
        self.noise = OrnsteinUhlenbeckActionNoise(self.config.reduced_action_dim)


        self.modules = [('actor', self.actor), ('Q', self.Q), ('action_rep', self.action_rep),
                        ('target_actor', self.target_actor), ('target_Q', self.target_Q)]#,
                        # ('target_action_rep', self.target_action_rep)]

        self.init()

    def get_action(self, state, explore=0):
        if self.batch_norm: self.actor.eval()  # Set the actor to Evaluation mode. Required for Batchnorm

        if self.initial_phase:
            # take random actions (uniformly in actual action space) to observe the interactions initially
            action = np.random.randint(self.action_dim)
            action_emb = self.action_rep.get_embedding(action).cpu().view(-1).data.numpy()

        else:
            state = tensor(state, dtype=float32, requires_grad=False, device=self.config.device).view(1, -1)
            action_emb = self.actor.get_action(state)

            noise = self.noise.sample() #* 0.1
            action_emb += Variable(torch.from_numpy(noise).type(float32), requires_grad=False)

            action = self.action_rep.get_best_match(action_emb)
            action_emb = action_emb.cpu().view(-1).data.numpy()

        self.track_entropy_cont(action_emb)
        return action, action_emb

    def update(self, s1, a1, a_emb1, r1, s2, done):
        self.memory.add(s1, a1, a_emb1, r1, s2, int(done != 1))
        if self.initial_phase and self.memory.length >= self.config.buffer_size:
            self.initial_phase_training(max_epochs=self.config.initial_phase_epochs)
        elif not self.initial_phase and self.memory.length > self.config.sup_batch_size:
            self.optimize()

    def optimize(self):
        if self.batch_norm: self.actor.train()  # Set the actor to training mode. Required for Batchnorm

        s1, a1, a1_emb, r1, s2, not_absorbing = self.memory.sample(self.config.sup_batch_size)

        # print(s1.shape, a1.shape, a1_emb.shape, r1.shape, s2.shape, not_absorbing.shape)
        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        a2_emb = self.target_actor.get_action(s2).detach()                      # Detach targets from grad computation.
        next_val = self.target_Q.forward(s2, a2_emb).detach()                   # Compute Q'( s2, pi'(s2))
        val_exp  = r1 + self.config.gamma * next_val * not_absorbing           # y_exp = r + gamma * Q'( s2, pi'(s2))

        val_pred = self.Q.forward(s1, a1_emb)                   # y_pred = Q( s1, a1)
        # loss_Q = F.smooth_l1_loss(val_pred, val_exp)                    # compute critic loss
        loss_Q = F.mse_loss(val_pred, val_exp)
        self.Q.update(loss_Q)

        # ---------------------- optimize actor ----------------------
        pred_a1_emb = self.actor.get_action(s1)
        loss_actor = -1.0 * torch.mean(self.Q.forward(s1, pred_a1_emb))
        self.actor.update(loss_actor)

        # ------------ update target actor and critic -----------------
        soft_update(self.target_actor, self.actor, self.config.tau)
        soft_update(self.target_Q, self.Q, self.config.tau)

        if not self.config.true_embeddings and self.config.emb_lambda > 0:
            self.ctr += 1
            if self.ctr > 100:
                self.self_supervised_training()
                self.ctr = 0


    def self_supervised_training(self, eps=1e-3):
        prv_loss = 1e5
        while True:
            s1, a1, _, _, s2, _ = self.memory.sample(batch_size=self.config.sup_batch_size)
            loss = self.action_rep.unsupervised_loss(s1, a1.view(-1), s2)
            self.action_rep.update(loss)
            # soft_update(self.target_action_rep, self.action_rep, self.config.tau)

            # quick check for convergence, break
            loss = loss.item()
            if prv_loss - loss < eps:
                break

            prv_loss = loss


    def initial_phase_training(self, max_epochs=-1):
        if self.batch_norm: self.actor.train()  # Set the actor to training mode. Required for Batchnorm

       # change optimizer to Adam for unsupervised learning
        self.action_rep.optim = torch.optim.Adam(self.action_rep.parameters(), lr=1e-3)
        initial_losses = []

        print("Inital training phase started...")
        for counter in range(max_epochs):
            losses = []
            for s1, a1, _, _, s2, _ in self.memory.batch_sample(batch_size=self.config.sup_batch_size,
                                                                randomize=True):
                loss_act_rep = self.action_rep.unsupervised_loss(s1, a1.view(-1), s2)
                self.action_rep.update(loss_act_rep)
                losses.append(loss_act_rep.item())

            initial_losses.append(np.mean(losses))
            if counter % 1 == 0:
                print("Epoch {} loss:: {}".format(counter, np.mean(initial_losses[-10:])))
                if self.config.only_phase_one:
                    self.save()
                    print("Saved..")

            # Terminate initial phase once action representations have converged.
            if len(initial_losses) >= 20 and np.mean(initial_losses[-10:]) + 1e-5 >= np.mean(initial_losses[-20:]):
                print("Converged...")
                break

        # Reset the optim to whatever is there in config
        self.action_rep.optim = self.config.optim(self.action_rep.parameters(), lr=self.config.embed_lr)

        print('... Initial training phase terminated!')
        self.initial_phase = False
        self.save()

        if self.config.only_phase_one:
            exit()

        hard_update(self.target_action_rep, self.action_rep)