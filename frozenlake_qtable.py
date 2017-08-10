# See https://gym.openai.com/envs/FrozenLake-v0 for details of the Gym env

import gym
import numpy as np

env = gym.make('FrozenLake-v0')
env.render()

def manualReset():
    env.reset()

def manualStep(action):
    _,_,_,_ = env.step(action)
    env.render()

def randomActionSampling(num_episodes=20):
    for i_episode in range(num_episodes):
        observation = env.reset()
        for t in range(100):
            env.render()
            # print(observation)
            action = env.action_space.sample()
            print(action)
            observation, reward, done, info = env.step(action)
            env.render()
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

def qLearning(num_episodes=1000):
    # init Q-table
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    lr = .8
    gamma = .95
    # num_episodes = 1000

    rewardsList = []

    for i in range(num_episodes):
        # reset environment to get new start state
        s = env.reset()
        rTotalEpisode = 0
        goalReachedBool = False
        # j = 0
        # Q-learning (sample-based Q-value iteration
        # while j < 99:
            # j+=1
        while (not goalReachedBool):
            # choose an action by greedily with noise picking from Q-table, decreasing randomness over time (as learning has made the Q-table more informative)
            a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
            # env.render()
            # epsilon greedy action selection
            # get new state and reward after taken the above a in the env
            s_next,r,goalReachedBool,_ = env.step(a)
            # update Q-table based on this experience (temporal difference learning)
            Q[s,a] = Q[s,a] + lr * (r + gamma*np.max(Q[s_next,:]) - Q[s,a])
            rTotalEpisode += r
            s = s_next
            # if goalReachedBool == True:
            #     break
        rewardsList.append(rTotalEpisode)

    print("Average reward after %i episodes: %0.2f" %(num_episodes,(sum(rewardsList)/num_episodes)))
    # print("Env view:")
    # print(env.render())
    # print("Final Q-table values:")
    # print(Q)

    print("Optimal policy from these Q-values:")
    print(np.argmax(Q,1))

# randomActionSampling()
# num_episodes 10k yields aver. reward of .70 >> clearly a more clever space exploration is needed (e.g. exploration fnc?)
qLearning(2000)

# manualReset()
# for t in range(20):
#     act=0
#     manualStep(act)