from __future__ import division

import dqn
import gym
import numpy as np
import random

# import matplotlib.pyplot as plt
import scipy.misc
import os

from gridworld import gameEnv
env = gameEnv(partial=False, size=5)

print('bal')

testMnih = dqn.QnetworkMnih13()
testMnih.runTraining(env)


