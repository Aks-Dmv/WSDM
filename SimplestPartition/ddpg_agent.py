import numpy as np
import random
import copy
from collections import namedtuple, deque
from replay_buffer import ReplayBuffer

import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Input, Add, Activation, LeakyReLU, Dropout
from keras.layers import GaussianNoise, Concatenate, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import copy
from rep import Memory

NUM_DIM=2
NUM_ACTIONS = NUM_DIM+2
TREE_DEPTH=9
BUFFER_SIZE = int(2*1e3)  # replay buffer size
BATCH_SIZE = 64        # minibatch size
GAMMA = 0.9            # discount factor
TAU = 5*1e-2              # for soft update of target parameters
LR_ACTOR = 2*1e-3         # learning rate of the actor
LR_CRITIC = 1e-2        # learning rate of the critic

EPSILON_DECAY = 2000
FINAL_EPSILON = 0.1
INITIAL_EPSILON = 0.3

# Fully Connected Layer's size was set to
# 2*NUM_ACTIONS
FC_ACTOR = 64*NUM_ACTIONS
FC_CRITIC = 64*NUM_ACTIONS


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self,env, sess, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.env = env
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.epsilon=INITIAL_EPSILON

        self.sigmaParam = 0.5



        # Actor Network (w/ Target Network)
        self.actor_local = self.Actor(state_size, action_size, random_seed,FC_ACTOR)
        self.actor_target = self.Actor(state_size, action_size, random_seed,FC_ACTOR)
        self.actor_local.compile(loss='mse', optimizer=Adam(lr=LR_ACTOR))

        # where we will feed de/dC (from critic)
        self.actor_state_input = self.actor_local.input
        # The line below means that we there will be rows (batch size in number)
        # and columns (action_size in number) where each i,j means the ith sample's
        # jth gradient
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, action_size])
        self.actor_local_weights = self.actor_local.trainable_weights

        # dC/dA (from actor)
        self.actor_grads = tf.gradients(self.actor_local.output, self.actor_local_weights, -self.actor_critic_grad)

        grads = zip(self.actor_grads, self.actor_local_weights)
        self.optimize = tf.train.AdamOptimizer(LR_ACTOR).apply_gradients(grads)
        # Critic Network (w/ Target Network)
        self.critic_local = self.Critic(state_size, action_size, random_seed,FC_CRITIC)
        self.critic_target = self.Critic(state_size, action_size, random_seed,FC_CRITIC)
        self.critic_local.compile(loss='mse', optimizer=Adam(lr=LR_CRITIC))

        #Critic Gradients
        self.critic_state_input,self.critic_action_input=self.critic_local.input
        self.critic_grads = tf.gradients(self.critic_local.output, self.critic_action_input)
        # Replay memory
        self.memory = Memory(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Initialize for later gradient calculations
        self.sess.run(tf.global_variables_initializer())

    def reset(self):
        print("agent has nothing to reset")

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        batch=copy.deepcopy(experiences)

        errors = self._getTargets( batch )
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])


        # for o in experiences:
        #     print("youas",o[1][0])
        # states = np.array([ o[1][0] for o in batch ])
        states = np.vstack([e[1][0] for e in experiences if e is not None])
        actions = np.vstack([e[1][1] for e in experiences if e is not None])
        rewards = np.vstack([e[1][2] for e in experiences if e is not None])
        next_states = np.vstack([e[1][3] for e in experiences if e is not None])
        dones = np.vstack([e[1][4] for e in experiences if e is not None])
        # states, actions, rewards, next_states, dones = experiences[:][1]

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.predict(next_states)
        Q_targets_next = self.critic_target.predict([next_states, actions_next])
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        loss = self.critic_local.train_on_batch([states,actions], Q_targets )

        # ---------------------------- update actor ---------------------------- #

        actions_pred = self.actor_local.predict(states)
        # Doubt: look into if the [0] at the end of grads
        #       is necessary

        # The below evaluates self.critic_grads
        # By feeding the inputs
        grads = self.sess.run(self.critic_grads, feed_dict={
            self.critic_state_input:  states,
            self.critic_action_input: actions_pred
        })[0]

        self.sess.run(self.optimize, feed_dict={
            self.actor_state_input: states,
            self.actor_critic_grad: grads
        })

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)




    def Critic(self, state_size, action_size, seed, fc_units=16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (int): Number of nodes in the hidden layers
        """

        S = Input(shape=[state_size])
        A = Input(shape=[action_size])

        S1 = Dense(fc_units)(S)
        S1 = GaussianNoise(0.1)(S1)
        S1 = BatchNormalization()(S1)
        S1 = Activation('relu')(S1)
        S1 = Dropout(0.5)(S1)

        S2 = Dense(fc_units)(S1)
        S2 = GaussianNoise(0.1)(S2)
        S2 = BatchNormalization()(S2)
        S2 = Activation('relu')(S2)
        S2 = Dropout(0.5)(S2)

        A1 = Dense(fc_units)(A)
        A1 = GaussianNoise(0.1)(A1)
        A1 = BatchNormalization()(A1)
        A1 = Activation('relu')(A1)
        A1 = Dropout(0.5)(A1)

        h0 = Add()([A1,S2])
        h0 = Dense(fc_units)(h0)
        h0 = GaussianNoise(0.1)(h0)
        h0 = BatchNormalization()(h0)
        h0 = Activation('relu')(h0)
        h0 = Dropout(0.5)(h0)

        h1 = Dense(int(fc_units/4) )(h0)
        h1 = GaussianNoise(0.1)(h1)
        h1 = BatchNormalization()(h1)
        h1 = Activation('relu')(h1)
        h1 = Dropout(0.5)(h1)

        finalOutput = Dense(1)(h1)
        finalOutput = BatchNormalization()(finalOutput)
        finalOutput = LeakyReLU(alpha=0.1)(finalOutput)
        return Model([S,A],finalOutput)


    def Actor(self, state_size, action_size, seed, fc_units=16):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (int): Number of nodes in the hidden layers

        """


        S = Input(shape=[state_size])
        h0 = Dense(fc_units)(S)
        h0 = GaussianNoise(0.1)(h0)
        h0 = BatchNormalization()(h0)
        h0 = Activation('relu')(h0)
        h0 = Dropout(0.5)(h0)

        h1 = Dense(fc_units)(h0)
        h1 = GaussianNoise(0.1)(h1)
        h1 = BatchNormalization()(h1)
        h1 = Activation('relu')(h1)
        h1 = Dropout(0.5)(h1)

        h1 = Dense(fc_units)(h1)
        h1 = GaussianNoise(0.1)(h1)
        h1 = BatchNormalization()(h1)
        h1 = Activation('relu')(h1)
        h1 = Dropout(0.5)(h1)

        Regressor = Dense(1)(h1)
        ActionToTake = Dense(action_size-1,activation='softmax')(h1)
        # Note: The policy has NUM_DIM + 1 in length
        #       The last node is for done action
        Policy=Concatenate()([ActionToTake,Regressor])


        return Model(S,Policy)




    def NetworkSummary(self):
        print("Actor Summary")
        self.actor_target.summary()
        print("Critic Summary")
        self.critic_target.summary()

    def load_network(self, path, extension):
        self.actor_local.load_weights(path+'actor_local_'+extension)
        self.actor_target.load_weights(path+'actor_target_'+extension)
        self.critic_local.load_weights(path+'critic_local_'+extension)
        self.critic_target.load_weights(path+'critic_target_'+extension)
        print("Successfully saved network.")

    def save_network(self, path, extension):
        # Saves model at specified extension as h5 file

        self.actor_local.save(path+'actor_local_'+extension)
        self.actor_target.save(path+'actor_target_'+extension)
        self.critic_local.save(path+'critic_local_'+extension)
        self.critic_target.save(path+'critic_target_'+extension)
        #print("Successfully saved network.")


    def _getTargets(self, batch):

        # batch=batch[1]
        no_state = np.zeros(2*NUM_DIM)
        #print("length of batch",batch)
        # print("length of batch",batch)
        #

        states = np.array([ o[1][0] for o in batch ])
        states_ = np.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ])


        #print(states)
        action1 = self.actor_local.predict(states)
        # print(action1)
        # print("the real sa ", [states, action1],len([states, action1]))
        p = self.critic_local.predict([states, action1])

        act1 = self.actor_local.predict(states_)
        p_ = self.critic_local.predict([states_, act1])

        act2 = self.actor_target.predict(states_)
        pTarget_ = self.critic_target.predict([states_, act2])

        errors = np.zeros(len(batch))

        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]; d=o[4]

            t = p[i]
            oldVal = t
            if s_ is None:
                t = r
            else:
                t = r + GAMMA * pTarget_[i]*(1-d)  # double DQN

            #print(r)
            #print(oldVal,t,GAMMA * pTarget_[i]*(1-d) )
            errors[i] = abs(oldVal - t)

        return errors

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        sample=(state, action, reward, next_state, done)
        errors = self._getTargets([[0, sample]])
        self.memory.add(errors[0], state, action, reward, next_state, done)

        # Slowly decay the learning rate
        if self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY

        # Learn, if enough samples are available in memory
        # if self.memory.tree.write > BATCH_SIZE:
        experiences = self.memory.sampleInternal()
        #print(experiences)
        self.learn(experiences, GAMMA)

    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def act(self, state,add_noise):
        """Returns actions for given state as per current policy."""
        state=np.array([state])
        #print("state ",state)
        #print("self.actor_local.predict(state)[0] ",self.actor_local.predict(state)[0])
        #print("self.actor_local.predict(state) ",self.actor_local.predict(state))
        action = self.actor_local.predict(state)[0]

        if add_noise:
            """
            If we are adding noise, we clearly know that we are exploring.
            """
            orig_weights=np.array( self.actor_local.get_weights() )
            al=np.random.normal(1., self.sigmaParam*self.epsilon )
            self.actor_local.set_weights( orig_weights*al )
            noisyAction = self.actor_local.predict(state)[0]
            self.actor_local.set_weights( orig_weights )
            action=noisyAction

            rand_val = np.random.random()
            if rand_val < self.epsilon:
                #print("if running")
                action[:-1] = np.random.random_sample( self.action_size-1 )
                action[:-1]=self.softmax(action[:-1])

        return action



    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        local_weights = local_model.get_weights()
        target_weights = target_model.get_weights()

        for i in range(len(local_weights)):
            target_weights[i] = tau * local_weights[i] + (1 - tau)* target_weights[i]

        target_model.set_weights(target_weights)
