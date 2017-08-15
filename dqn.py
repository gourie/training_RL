import gridworld
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim

class QnetworkMnih13():
    def __init__(self, env_numOfActions=4):
        self.conv_layers = 2
        self.nn_params = dict(learning_rate=0.0001)
        self.rl_params = dict(gamma=.99, epsilon=.1, num_episodes=2000, replay_batch_size=32, replay_buffer_size=5e5, replay_pretrain_size=3)
        self.conv_layer1 = dict(filters=16, filterSize=8, stride=4, nonlinear='relu')
        self.conv_layer2 = dict(filters=32, filterSize=4, stride=2, nonlinear='relu')
        self.conv_layer3 = dict(filters=64, filterSize=3, stride=1, nonlinear='relu')
        self.conv_layer4 = dict(filters=64, filterSize=7, stride=1, nonlinear='relu')
        self.fc_layer1 = dict(nodes=256, nonlinear='relu')
        self.fc_layer2 = dict(nodes=env_numOfActions, nonlinear='none')

        self.input_data = None
        self.Q = None
        self.best_action = None

        self.buildModel()

    def buildModel(self, inputSize=21168):
        """ Build regression modeol that minimizes using qTarget (satisfying Bellman eq using current weights) - predictQ, train using Adam to find optimal W
        Args:
            inputSize: length of list provided as input
        Returns:
             None, updates the list of variables collected in the graph under the key GraphKeys.TRAINABLE_VARIABLES.
        Raises:
            None
        """
        # model architecture
        self.input_data = tf.placeholder(shape=[None,inputSize], dtype=tf.float32)
        l1_output = tf.contrib.layers.conv2d(inputs=tf.reshape(self.input_data, shape=[-1, 84, 84, 3]), num_outputs=self.conv_layer1['filters'],
                                     kernel_size=[self.conv_layer1['filterSize'], self.conv_layer1['filterSize']],
                                     stride=[self.conv_layer1['stride'],self.conv_layer1['stride']], padding='VALID', activation_fn=tf.nn.relu, scope='CONV1')
        l2_output = tf.contrib.layers.conv2d(inputs=l1_output, num_outputs=self.conv_layer2['filters'],
                                     kernel_size=[self.conv_layer2['filterSize'], self.conv_layer2['filterSize']],
                                     stride=[self.conv_layer2['stride'],self.conv_layer2['stride']], padding='VALID', activation_fn=tf.nn.relu, scope='CONV2')
        l3_output = tf.contrib.layers.conv2d(inputs=l2_output, num_outputs=self.conv_layer3['filters'],
                                     kernel_size=[self.conv_layer3['filterSize'], self.conv_layer3['filterSize']],
                                     stride=[self.conv_layer3['stride'],self.conv_layer3['stride']], padding='VALID', activation_fn=tf.nn.relu, scope='CONV3')
        l4_output = tf.contrib.layers.conv2d(inputs=l3_output, num_outputs=self.conv_layer4['filters'],
                                     kernel_size=[self.conv_layer4['filterSize'], self.conv_layer4['filterSize']],
                                     stride=[self.conv_layer4['stride'],self.conv_layer4['stride']], padding='VALID', activation_fn=tf.nn.relu, scope='CONV4')
        l5_output = tf.contrib.layers.fully_connected(inputs=l4_output, num_outputs=self.fc_layer1['nodes'],
                                     activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), scope='FC1')

        self.Q = tf.contrib.layers.fully_connected(inputs=l5_output, num_outputs=self.fc_layer2['nodes'],
                                     activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(), scope='FC2')
        self.best_action = tf.argmax(self.Q, axis=3, name="bestAction")

        # loss and optimizer
        q_target = tf.placeholder(shape=[None], dtype=tf.float32)
        reg_loss = tf.reduce_mean(tf.square(q_target - self.Q))
        tf.train.AdamOptimizer(self.nn_params['learning_rate']).minimize(reg_loss)

    def runTraining(self, env):

        # init NN model params
        init = tf.global_variables_initializer()
        # create lists to contain total steps and rewards per episode
        jList = []
        rList = []
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./graphs', sess.graph)
            sess.run(init)
            memoryBuffer = ExperienceBuffer(buffer_size=self.rl_params['replay_buffer_size'])

            for i in range(self.rl_params['num_episodes']):

                s = gridworld.processState(env.reset())   # specific to gridWorld
                endOfEpisode = False
                rAll = 0

                # j = 0
                # # The Q-Network
                # while j < 99:
                #     j += 1  #control nb of steps in episode
                while (not endOfEpisode):

                    bool_pretrain_to_fill_memory_buffer = (memoryBuffer.getActualSize() < self.rl_params['replay_pretrain_size'])
                    # take epsilon-greedy action (exploration at start of RL training!!)
                    if (np.random.rand(1) < self.rl_params['epsilon']) or bool_pretrain_to_fill_memory_buffer:
                        # GYM
                        # a = env.action_space.sample()
                        a = np.random.randint(0,4)
                    else:
                        a,q = sess.run([self.best_action,self.Q], feed_dict={self.input_data:[s]})
                        a = a[0]

                    # execute action in env
                    s_next, r, endOfEpisode = env.step(a)
                    s_next = gridworld.processState(s_next)   # specific to gridWorld

                    # store transition into replay memory
                    memoryBuffer.add(np.reshape(np.array([s,a,r,s_next,endOfEpisode]), [1, 5]))

                    # once the replay memory buffer has sufficient experiences, start training!
                    if (not bool_pretrain_to_fill_memory_buffer):
                        print("Replay buffer filled with %i samples, lets's start training the model!" %(self.rl_params['replay_pretrain_size']))
                        # take sample from replay memory and train NN
                        # mini_batch = memoryBuffer.sample(self.rl_params['replay_batch_size'])

                        # Reduce chance of random action as we train the model.
                        # self.rl_params['epsilon'] = 1. / ((i / 50) + 10)

                    if endOfEpisode == True:
                        break




class QnetworkJuliani():
    def __init__(self, h_size, env_numOfActions):
        # The network receives a frame from the game, flattened into an array of size 21168.
        # The flattened array is resized to a 84x84x3 (rgb channels) and processed by four convolutional layers.
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])
        self.conv1 = slim.conv2d( \
            inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding='VALID',
            biases_initializer=None)
        self.conv2 = slim.conv2d( \
            inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding='VALID',
            biases_initializer=None)
        self.conv3 = slim.conv2d( \
            inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='VALID',
            biases_initializer=None)
        self.conv4 = slim.conv2d( \
            inputs=self.conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1], padding='VALID',
            biases_initializer=None)

        # We take the output from the final convolutional layer and split it into separate advantage and value streams along dim 3 (h_size).
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size // 2, env_numOfActions]))
        self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env_numOfActions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

#TODO(joerinicolaes@gmail.com): use python heap queue for more effective FIFO queue
class ExperienceBuffer():
    """
    Experience_buffer class to implement experience replay inside DQN
    """
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def getActualSize(self):
        return len(self.buffer)

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            # if buffer gets full, remove old experiences (start of buffer) > FIFO concept
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, nb_of_samples):
        """Uniform sampling of nb_of_samples from the experience buffer

        Returns nb_of_samples experiences from buffer, each experience consisting of tuple
        (s,a,r,next_s,bool_terminal_state)

        Args:
            nb_of_samples: an int specifying how many experience samples must be returned
        Returns:
            A numpy array reshaped to shape (nb_of_samples,5)
        Raises:
            None
        """
        return np.reshape(np.array(random.sample(self.buffer, nb_of_samples)), [nb_of_samples, 5])