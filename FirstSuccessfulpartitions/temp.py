from replay_buffer import ReplayBuffer
from SumTree import SumTree
from rep import Memory
import numpy as np
NUM_DIM=2
NUM_ACTIONS = NUM_DIM+2
TREE_DEPTH=8
BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 3
action_size=4
random_seed=67

state=np.array([[-1.,4.,1.,6.]])
action=np.array([[0.2,0.3,0.5,6.]])
reward=6
done=False
next_state=np.array([[-1.,4.,1.,6.]])

memory1 = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
memory1.add(state, action, reward, next_state, done)
memory1.add(state, action, reward, next_state, done)
memory1.add(state, action, reward, next_state, done)
memory1.add(state, action, reward, next_state, done)
memory1.add(state, action, reward, next_state, done)


memory = Memory(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
memory.add(0.3,state, action, reward, next_state, done)
memory.add(0.4,state, action, reward, next_state, done)
memory.add(0.1,state, action, reward, next_state, done)
memory.add(0.5,state, action, reward, next_state, done)
memory.add(0.6,state, action, reward, next_state, done)

print(memory1.sample())
print(memory.sample())
# experiences=memory.sample()
# states = np.vstack([e[1][0] for e in experiences if e is not None])
# actions = np.vstack([e[1][1] for e in experiences if e is not None])
# rewards = np.vstack([e[1][2] for e in experiences if e is not None])
# nstates = np.vstack([e[1][3] for e in experiences if e is not None])
# print("states")
# print((states,actions,rewards,nstates))
# #print(actions)
