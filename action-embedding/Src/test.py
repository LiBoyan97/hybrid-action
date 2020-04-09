import numpy as np
from Src.Utils.utils import Space, binaryEncoding
import torch
action_dim=10
reduced_action_dim=2
action = np.random.randint(10)
# print(action)
init_tensor = torch.rand(action_dim, reduced_action_dim)*2 - 1
# print(init_tensor)
embeddings = torch.nn.Parameter(init_tensor, requires_grad=True)
action_emb = embeddings[action]
# print(embeddings)
# print(action_emb)
n_actions=2
shape = (2**n_actions, 2)
print(shape[0])
motions = np.zeros(shape)
motions[0]=[1,2]
print(motions)
