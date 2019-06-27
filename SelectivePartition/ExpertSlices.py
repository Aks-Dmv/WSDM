from collections import namedtuple, deque
import random
import numpy as np
import copy

class ExpertSlices:
    """Interacts with and learns from the environment."""

    def __init__(self, buffer_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        # The range is the max variation of the other independent
        # variables
        self.experience = namedtuple("Experience", field_names=["Dim", "Val", "Range"])
        self.seed = random.seed(seed)
        self._trueBoundaries=np.array([[-1.,4.],[1.,6.]])
        self._maxRegressVal=2


    def add(self, dim, val, range):
        """Add a new experience to memory."""
        e = self.experience(dim, val, range)
        self.memory.append(e)

    def action(self, state):
        # State will be of the form
        # np.array([[-1.,4.],[1.,6.]])
        #print("state ",state,"\n")
        actions=[]
        for ele in self.memory:
            # If the val lies in between the dims
            #print("ele is ",ele)
            #print(state)
            actionMultiFactor=(self._trueBoundaries[int(ele.Dim)][1]-self._trueBoundaries[int(ele.Dim)][0])/self._maxRegressVal
            temp=ele.Val*actionMultiFactor+self._trueBoundaries[int(ele.Dim)][0]

            # If the value is in between the state bounds
            if(temp>state[int(ele.Dim)][0] and temp<state[int(ele.Dim)][1]):
                #print("ele.Val and state",ele.Val,state[int(ele.Dim)])
                remainingPerimeter=np.delete(copy.deepcopy(state),int(ele.Dim),0)
                check=True
                for i in range(len(remainingPerimeter)):
                    # the state is enclosed in the perimeter if
                    enclosed=ele.Range[i][1]>=remainingPerimeter[i][1] and ele.Range[i][0]<=remainingPerimeter[i][0]
                    if(not enclosed):
                        check=False

                if(check):
                    actions.append([ele.Dim,ele.Val])
        #print("Expert action set for state ",state,"is ",actions)
        return actions

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
