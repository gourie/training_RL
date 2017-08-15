import gym
import numpy as np
import random
import tensorflow as tf
# import matplotlib.pyplot as plt
# %matplotlib inline

env = gym.make('CartPole-v0')
print(env.observation_space.n)
print(env.state_space.n)
env.render()
