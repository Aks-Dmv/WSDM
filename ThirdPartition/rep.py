import random, numpy, math, scipy
from SumTree import SumTree



LEARNING_RATE = 0.00025


#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.tree = SumTree(buffer_size)
        self.action_size = action_size
        self.max_size=buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, state, action, reward, next_state, done):
        sample=(state, action, reward, next_state, done)
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sampleInternal(self):
        n=self.batch_size
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def sample(self):
        n=self.batch_size
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

            states = numpy.vstack([e[1][0] for e in batch if e is not None])
            actions = numpy.vstack([e[1][1] for e in batch if e is not None])
            rewards = numpy.vstack([e[1][2] for e in batch if e is not None])
            nstates = numpy.vstack([e[1][3] for e in batch if e is not None])
            dones = numpy.vstack([e[1][4] for e in batch if e is not None])

        return (states,actions,rewards,nstates,dones)

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


    #
    #
    #
    #
    #
    # def observe(self, sample):  # in (s, a, r, s_) format
    #     x, y, errors = self._getTargets([(0, sample)])
    #     self.memory.add(errors[0], sample)
    #
    #     if self.steps % UPDATE_TARGET_FREQUENCY == 0:
    #         self.brain.updateTargetModel()
    #
    #     # slowly decrease Epsilon based on our eperience
    #     self.steps += 1
    #     self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
    #
    # def _getTargets(self, batch):
    #     no_state = numpy.zeros(self.stateCnt)
    #
    #     states = numpy.array([ o[1][0] for o in batch ])
    #     states_ = numpy.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ])
    #
    #     p = agent.brain.predict(states)
    #
    #     p_ = agent.brain.predict(states_, target=False)
    #     pTarget_ = agent.brain.predict(states_, target=True)
    #
    #     x = numpy.zeros((len(batch), IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT))
    #     y = numpy.zeros((len(batch), self.actionCnt))
    #     errors = numpy.zeros(len(batch))
    #
    #     for i in range(len(batch)):
    #         o = batch[i][1]
    #         s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
    #
    #         t = p[i]
    #         oldVal = t[a]
    #         if s_ is None:
    #             t[a] = r
    #         else:
    #             t[a] = r + GAMMA * pTarget_[i][ numpy.argmax(p_[i]) ]  # double DQN
    #
    #         x[i] = s
    #         y[i] = t
    #         errors[i] = abs(oldVal - t[a])
    #
    #     return (x, y, errors)
    #
    # def learn(self):
    #     batch = self.memory.sample(BATCH_SIZE)
    #     x, y, errors = self._getTargets(batch)
    #
    #     #update errors
    #     for i in range(len(batch)):
    #         idx = batch[i][0]
    #         self.memory.update(idx, errors[i])
    #
    #     self.brain.train(x, y)
