from util import ReplayBuffer
from dqn import DeepQNetwork

class DQNAgent():
  def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size,
               batch_size, eps_min = 0.01, eps_dec = 5e-7,
               replace = 1000, algo = None, env_name = None, chkpt_dir = 'tmp/dqn'):

    self.gamma = gamma
    self.epsilon = epsilon
    self.lr = lr
    self.n_actions = n_actions
    self.input_dims = input_dims
    self.batch_size = batch_size
    self.eps_min = eps_min
    self.eps_dec = eps_dec
    self.replace_target_cnt = replace
    self.algo = algo
    self.env_name = env_name
    self.chkpt_dir = chkpt_dir
    self.action_space = [i for i in range(self.n_actions)]
    self.learn_step_counter = 0 # learning trigger

    self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

    self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                               input_dims= self.input_dims,
                               name = self.env_name + '_' + self.algo + '_q_eval',
                               chkpt_dir = self.chkpt_dir)

    # backpop never happens in q_next - we only use this for target
    self.q_next = DeepQNetwork(self.lr, self.n_actions,
                            input_dims= self.input_dims,
                            name = self.env_name + '_' + self.algo + '_q_next',
                            chkpt_dir = self.chkpt_dir)

  def choose_action(self, state):
    if np.random.random() > self.epsilon: # exploration
      action = np.random.choice(self.action_space)
    else: # exploitation
      # state is in a list i.e. [state] because NN network takes inputs of the form batch size x input_dims
      state = T.tensor([state], dtype=T.float).to(self.q_eval.device)
      Q_values = self.q_eval.forward(state)
      action = T.argmax(Q_values).item() # to numpy array
    return action

  def store_transition(self, state, action, reward, state_, done):
    self.memory.store_transition(state, action, reward, state_, done)

  def sample_memory(self):
    state, action, reward, new_state, done = \
        self.memory.sample_buffer(self.batch_size)

    states = T.tensor(state).to(self.q_eval.device)
    rewards = T.tensor(reward).to(self.q_eval.device)
    dones = T.tensor(done).to(self.q_eval.device)
    actions = T.tensor(action).to(self.q_eval.device)
    states_ = T.tensor(new_state).to(self.q_eval.device)

    return states, rewards, actions, states_, dones

  def replace_target_network(self):
    if self.learn_step_counter % self.replace_target_cnt == 0:
      self.q_next.load_state_dict(self.q_eval.state_dict()) # update weights of target network the same with the behavior network

  def decrement_epsilon(self):
    self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

  def save_models(self):
    self.q_eval.save_checkpoint()
    self.q_next.save_checkpoint()

  def load_models(self):
    self.q_eval.load_checkpoint()
    self.q_next.load_checkpoint()

  def learn(self):
  # update Q values here
    if(self.memory.mem_cntr < self.batch_size):
        return

    self.q_eval.optimizer.zero_grad()

    self.replace_target_network()

    states, rewards, actions, states_, dones = self.sample_memory()

    indices = np.arange(self.batch_size)
    q_pred = self.q_eval.forward(states)[indices, actions]

    # self.q_eval.forward(states) --> batch_size x n_actions = 32 x 6

    #q_pred = self.q_eval.forward(states)[actions]
    # this is wrong because we only want q values for the action that the agent actually took in that state
    # see example below

    q_next = self.q_next.forward(states_).max(dim = 1)[0] # maximum of expected future reward

    q_next[dones] = 0.0

    q_target = rewards + self.gamma * q_next

    loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device) # difference between target and current Q values
    loss.backward() # backpropagate
    self.q_eval.optimizer.step() # update weights

    self.decrement_epsilon()
