import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
class DeepQNetwork(nn.Module):

  def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
    super(DeepQNetwork, self).__init__()

    self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride = 4) #1 input image channel, 6 output channels, 5x5 square convolution
    self.conv2 = nn.Conv2d(32, 64, 4, stride = 2)
    self.conv3 = nn.Conv2d(64, 64, 3, stride = 1)
    fc_input_dims = self.calculate_conv_output_dims(input_dims)

    self.fc1 = nn.Linear(fc_input_dims, 512) # input_size, action_space_size
    self.fc2 = nn.Linear(512, n_actions)

    self.optimizer = optim.RMSprop(self.parameters(), lr = lr)
    self.loss = nn.MSELoss()
    self.device = T.device(T.device('cuda'))
    self.to(self.device)

    self.checkpoint_dir = chkpt_dir
    self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

  def calculate_conv_output_dims(self, input_dims):
    state = T.zeros(1, *input_dims)
    dims = self.conv1(state)
    dims = self.conv2(dims)
    dims = self.conv3(dims)
    return int(np.prod(dims.size()))

  def forward(self, state):
    layer1 = F.relu(self.conv1(state))
    layer2 = F.relu(self.conv2(layer1))
    layer3 = F.relu(self.conv3(layer2))

    # conv3 shape is batch size x n_filters x H x W
    conv_state = layer3.view(layer3.size()[0], -1)
    flat1 = F.relu(self.fc1(conv_state))
    actions = self.fc2(flat1)

    return actions

  def save_checkpoint(self):
    print('... saving checkpoint ...')
    T.save(self.state_dict(), self.checkpoint_file)

  def load_checkpoint(self):
    print('... loading checkpoint ...')
    self.load_state_dict(T.load(self.checkpoint_file))
