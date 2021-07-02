import numpy as np
import matplotlib.pyplot as plt
import cv2
import collections

def plot_learning_curve(x, scores, epsilons, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig(filename)

# custom wrapper where we can change the outputs of step and reset
class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=0,
                 fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        t_reward = 0.0
        done = False
        # since we are using the same action 4 repeat times, we add the total rewards for those 4 frames
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info

    def reset(self):
        obs = self.env.reset()
        no_ops = np.random.randint(self.no_ops)+1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _ = self.env.step(1)

        self.frame_buffer = np.zeros_like((2,self.shape))
        self.frame_buffer[0] = obs

        return obs

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1]) # open AI returns channels last
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                    shape=self.shape, dtype=np.float32)

    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0

        return new_obs

class StackFrames(gym.ObservationWrapper):
  def __init__(self, env, repeat): # repeat here is the number of frames to repeat
    super(StackFrames, self).__init__(env)
    self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, axis = 0), # W x H x 4
                                            env.observation_space.high.repeat(repeat, axis = 0), # W x H x 4
                                            dtype=np.float32)
    self.stack = collections.deque(maxlen=repeat)

  def reset(self):
    self.stack.clear()
    observation = self.env.reset()
    for _ in range(self.stack.maxlen):
      self.stack.append(observation)
    return np.array(self.stack).reshape(self.observation_space.low.shape) # can use high here

  def observation(self, observation):
    self.stack.append(observation)
    return np.array(self.stack).reshape(self.observation_space.low.shape)

class ReplayBuffer(object):
  def __init__(self, max_size, input_shape, n_actions):
    self.mem_size = max_size
    self.mem_cntr = 0
    self.state_memory = np.zeros((self.mem_size, *input_shape),
                                 dtype=np.float32)
    self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
    self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
    self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
    self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

  def store_transition(self, state, action, reward, state_, done):
    # store the memories in the position of the first unoccupied memory
    index = self.mem_cntr % self.mem_size
    self.state_memory[index] = state
    self.action_memory[index] = action
    self.reward_memory[index] = reward
    self.new_state_memory[index] = state_
    self.terminal_memory[index] = done
    self.mem_cntr += 1

  def sample_buffer(self, batch_size):
    max_mem = min(self.mem_cntr, self.mem_size) # position of the last stored memory
    batch = np.random.choice(max_mem, batch_size, replace = False)

    states = self.state_memory[batch]
    actions = self.action_memory[batch]
    rewards = self.reward_memory[batch]
    states_ = self.new_state_memory[batch]
    dones = self.terminal_memory[batch]

    return states, actions, rewards, states_, dones
