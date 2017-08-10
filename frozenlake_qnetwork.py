import gym
import numpy as np
import random
import tensorflow as tf
# import matplotlib.pyplot as plt
# %matplotlib inline

env = gym.make('FrozenLake-v0')
print(env.observation_space.n)

# create tf graph
#1. forward calc
stateInp = tf.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4], 0, 0.01))
qOut = tf.matmul(stateInp, W)
bestAction = tf.argmax(qOut,1)

#2. backprop from Euclidean loss
qSample = tf.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(qSample - qOut))
sgd = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = sgd.minimize(loss)

init = tf.global_variables_initializer()

#3. start session to execute on the graph
y = .99
e = 0.1
num_episodes = 2000
#create lists to contain total steps and rewards per episode
jList = []
rList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while j < 99:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            a,allQ = sess.run([bestAction,qOut],feed_dict={stateInp:np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a[0])
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(qOut,feed_dict={stateInp:np.identity(16)[s1:s1+1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={stateInp:np.identity(16)[s:s+1],qSample:targetQ})
            rAll += r
            s = s1
            if d == True:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
        jList.append(j)
        rList.append(rAll)
print("Average reward after %i episodes: %0.2f" %(num_episodes,(sum(rList)/num_episodes)))
    # ("Percent of successful episodes: " + str(sum(rList)/num_episodes) + "%")