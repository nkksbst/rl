import gym
from dqn_agent import DQNAgent
from util import plot_learning_curve
from util import plot_learning_curve, RepeatActionAndMaxFrame, PreprocessFrame, StackFrames
import numpy as np
import os
import torch.nn as nn
src_dir = ''
def make_env(env_name, shape=(84,84,1), repeat = 4, clip_rewards = False, no_ops=0, fire_first=False):
  # no-ops : number of operations to skip during test mode

  env = gym.make(env_name)
  env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
  env = PreprocessFrame(shape, env)
  env = StackFrames(env, repeat)
  return env

def main():
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    
    load_checkpoint = False
    
    n_games = 500  # for training
    
    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                      input_dims=(env.observation_space.shape),
                      n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                      batch_size=32, replace=1000, eps_dec=1e-5,
                      chkpt_dir= src_dir + 'models', algo='DQNAgent',
                      env_name='PongNoFrameskip-v4')
    
    if load_checkpoint:
      agent.load_models()
    
    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + \
      '_' + str(n_games) + 'games'
    
    
    figure_file = src_dir + 'plots/' + fname + '.png'
    
    n_steps = 0
    scores, eps_history, steps_array = [], [], []
    
    for i in range(n_games):
      done = False
      score = 0
      observation = env.reset()
    
      while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
    
        if not load_checkpoint:
          agent.store_transition(observation, action, reward, observation_, done) # (4,84,84) int float (4,84,84) bool
          agent.learn()
    
        observation = observation_
        n_steps += 1
      scores.append(score)
      steps_array.append(n_steps)
    
      avg_score = np.mean(scores[-100:])
      print('episode ', i, 'score: ', score,
            'average score %.1f best score %.1f epsilon %.2f' %
            (avg_score, best_score, agent.epsilon),
            'steps ', n_steps)
      if avg_score > best_score:
        if not load_checkpoint:
          agent.save_models()
        best_score = avg_score
    
      eps_history.append(agent.epsilon)
    
    plot_learning_curve(steps_array, scores, eps_history, figure_file)

if __name__=='__main__':
    main()