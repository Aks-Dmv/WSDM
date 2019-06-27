import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
import copy
from infoCalc import *
from collections import namedtuple, deque
import random


class BFSWorldEnv:
    """
    Define a simple BFSWorld environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """


    def __init__(self):
        # General variables defining the environment
        self._trueDf=pd.read_csv("../data/NewdataPts.csv")
        self._trueBoundaries=np.array([[-1.,4.],[1.,6.]])

        # Our ouput should be in the form of
        # (N output variables, one stop variable) and one regressor variable
        """
        Don't remove the following because they are essential for
        other programs
        """
        SoftMaxOutput = np.tile( [0.,1.], (len(self._trueBoundaries)+1,1) )
        Regr=np.array([[self._trueBoundaries.min(), self._trueBoundaries.max()]])
        self.action_space = np.concatenate((SoftMaxOutput, Regr), axis=0)
        ob1 = copy.deepcopy(self._trueBoundaries)
        self.observation_space = np.reshape(ob1,(1,4))[0]

        # There are two queues. One holds to be processed states and the
        # other holds finished state action pairs (used for rendering)
        self.max_size = 1000
        self.StateBuffer = deque(maxlen=self.max_size)
        self.StateActionMemory = deque(maxlen=self.max_size)
        self.StateActionExp = namedtuple("StateAction", field_names=["state", "action"])
        self.UpcomingState = namedtuple("State", field_names=["state", "inheritedN"])

        # This is the first state to be processed
        self.addState( copy.deepcopy(self._trueBoundaries), 0 )

        self._infoRewardMultiplier=20
        self.penalty=-1
        self.OutOfBounds=-5
        # the max value the network's regress var should take
        self._maxRegressVal=2
        # For how many points is considered useless
        self.thresholdNumber=10

    def close(self):
        self.reset()
        print("Simulation Over")

    def NumberOfPartitions(self,thresh):
        # print(len(self.StateActionMemory))
        return len(self.StateActionMemory)>thresh

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.StateBuffer.clear()
        self.StateActionMemory.clear()
        self.addState( copy.deepcopy(self._trueBoundaries), 0 )
        return self.returnState(copy.deepcopy(self._trueBoundaries))

    def addStateBack(self, state, inheritedN):
        """Add an experience to back to the beginning of the queue memory.
            The state has to be in the _trueBoundaries format
        """
        e = self.UpcomingState(state, inheritedN)
        if len(self.StateBuffer) < self.max_size:
            self.StateBuffer.append(e)
        else:
            print("max Limit reached, Increase Buffer Size")


    def addState(self, state, inheritedN):
        """Add a new experience to memory.
            The state has to be in the _trueBoundaries format
        """
        e = self.UpcomingState(state, inheritedN)
        if len(self.StateBuffer) < self.max_size:
            self.StateBuffer.appendleft(e)
        else:
            print("max Limit reached, Increase Buffer Size")

    def addStateAction(self, state, action):
        """Add a new experience to memory.
            The state has to be in the _trueBoundaries format
        """
        e = self.StateActionExp(state, action)
        if len(self.StateActionMemory) < self.max_size:
            self.StateActionMemory.append(e)
        else:
            print("max Limit reached, Increase Buffer Size")

    def seed(self, s):
        #print("yo")
        random.seed(s)
        #print("yo")
        np.random.seed
        #print("yo")

    def returnState(self,state):
        ob = np.reshape(state,(1,len(self._trueBoundaries)*2))[0]
        return ob

    def normVal(self,val,dim):
        #the oppDim value has been hard coded
        # because the dataset is 2D
        oppDim=1-dim
        dimRange=self._trueBoundaries[oppDim][1]-self._trueBoundaries[oppDim][0]
        valNew=(val-self._trueBoundaries[oppDim][0])/dimRange

        return valNew

    def render(self):
        plt.scatter(self._trueDf['0'], self._trueDf['1'])
        #print(self.StateActionMemory)

        for i in self.StateActionMemory:
            # print("This is i",i)
            # print("this is the state",i[0][1][0])
            # print("this is the dimStop",i[1][:-1].argmax())
            # print("this is the val",i[1][-1])
            """
            i will look like
            StateAction(state=array([[-1.,  4.], [ 1.,  6.]]),
                action=array([0.2, 0.7, 0.1, 5. ]))

            the dimStop will either select the dimension or
            stop (if dimStop==len(self._trueBoundaries))

            self._trueBoundaries=np.array([[-1.,4.],[1.,6.]])
            """
            print(i)
            stateBounds = i[0]
            #print("state bounds",stateBounds)
            dimStop=i[1][:-1].argmax()
            val=i[1][-1]
            #print("action before this", i[1])
            # If the stop signal has been flagged
            if(dimStop==len(self._trueBoundaries)):
                continue
            actionMultiFactor=(self._trueBoundaries[dimStop][1]-self._trueBoundaries[dimStop][0])/self._maxRegressVal
            val=val*actionMultiFactor+self._trueBoundaries[dimStop][0]
            #print("going to be plotted ", val, dimStop)
            #print("val",val)
            #print(val)

            # if dimStop == 1 then we have choosen to partition @ val on the
            # x1th dimension, in our case y axis
            # thus, y coord is const and x changes ie. horizontal line
            if(dimStop==1):
                # the 1-dimStop is because we have to select the other dimension for plotting
                plt.axhline(y=val, xmin=self.normVal(stateBounds[1-dimStop][0],dimStop), xmax=self.normVal(stateBounds[1-dimStop][1],dimStop), color='r', linestyle='-')

            else:
                # the 1-dimStop is because we have to select the other dimension for plotting
                plt.axvline(x=val, ymin=self.normVal(stateBounds[1-dimStop][0],dimStop), ymax=self.normVal(stateBounds[1-dimStop][1],dimStop), color='r', linestyle='-')

        plt.title('Scatter plot')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show(block=False)
        #input("enter")
        plt.close()


    def step(self,action):

        softmaxOut=action[:-1].argmax()
        regrOut=action[-1]

        if(len(self.StateBuffer)==0):
            #print("game over")
            bounds = self.returnState(copy.deepcopy(self._trueBoundaries) )
            return bounds, 0,True,None

        #print(self.StateBuffer)


        if( softmaxOut==len(self._trueBoundaries) ):
            # If you are here, then the stop variable has been flagged
            bounds=self.StateBuffer.pop()[0]
            self.addStateAction( bounds, action )
            if( len(self.StateBuffer)==0 ):
                done=True
                bounds = self.returnState(copy.deepcopy(self._trueBoundaries) )
                # the bounds will be the same state because the environment froze
            else:
                done=False
                # The next state will be it's sibling
                bounds=self.StateBuffer[-1][0]
                bounds = self.returnState(copy.deepcopy(bounds) )

            """
            We should make the reward into the just self.penalty
            when the agent starts doing well
            We need an initial bias to kick things off
            # reward = self.penalty

            """
            #print(len(self.StateActionMemory))
            reward=20*self.penalty/(len(self.StateActionMemory)+1)
            # reward = self.penalty

            #reward=self._Quitpenalty #/(len(self.actionstatepairs)+1)
            # If the action taken is to stop, then we give zero reward

            return bounds, reward,done,None

        # If we have not returned yet, then that means the softmaxOut represents which dim
        dim=softmaxOut
        PresentStateAndN=self.StateBuffer.pop()
        PrState=PresentStateAndN[0]

        inheritedN=PresentStateAndN[1]
        actionMultiFactor=(self._trueBoundaries[dim][1]-self._trueBoundaries[dim][0])/self._maxRegressVal
        val=regrOut*actionMultiFactor+self._trueBoundaries[dim][0]
        #print("1val,self._trueBoundaries[dim][0],self._trueBoundaries[dim][1]",val,self._trueBoundaries[dim][0],self._trueBoundaries[dim][1])

        # to check if the regressor is out of bounds
        if(val<=PrState[dim][0]):
            lessTh=True
        else:
            lessTh=False
        if(val>=PrState[dim][1]):
            greaterTh=True
        else:
            greaterTh=False
        if(lessTh or greaterTh):
            # you exceeded the boundary
            # this includes selecting the boundary
            # Thus, your action was wrong
            # done is false because you just picked a wrong value
            # you can try again
            # note that the environment will not store this value
            # however, the S,A,R,S will be stored by the agent

            """
            We are making a rule that if you guess out of bounds,
            You lose a chance, the state doesn't change
            # self.addStateBack( PrState, inheritedN )

            After we popped, is the buffer empty
            """
            if( len(self.StateBuffer)==0 ):
                done=True
                # the bounds will be the same state because the environment froze
                bounds = self.returnState(copy.deepcopy(self._trueBoundaries) )
            else:
                done=False
                # The next state will be it's sibling
                bounds=self.StateBuffer[-1][0]
                bounds = self.returnState(copy.deepcopy(bounds) )
            # bounds=self.StateBuffer[-1][0]

            if(lessTh):
                # remember, this number is negative
                deltaBounds=1*(val-PrState[dim][0])+self.OutOfBounds
                #print("obo val, max/min dim", val,PrState[dim][0])

            if(greaterTh):
                deltaBounds=1*(PrState[dim][1]-val)+self.OutOfBounds
                #print("obo val, max/min dim", val,PrState[dim][1])

            return bounds, deltaBounds,done,None

        #print("2val,PrState[dim][0],PrState[dim][1]",val,PrState[dim][0],PrState[dim][1])



        df=self._trueDf[ (self._trueDf[str(0)] >= PrState[0][0]) & (self._trueDf[str(0)] <= PrState[0][1]) & (self._trueDf[str(1)] >= PrState[1][0]) & (self._trueDf[str(1)] <= PrState[1][1]) ]
        # Checking for a corner case

        """
        Number of points to consider as vacuum
        """

        if(len(df.index)<=self.thresholdNumber):
            # There are no elements
            # Thus, your action was futile

            # self.addStateBack( PrState, inheritedN )
            if( len(self.StateBuffer)==0 ):
                done=True
                # the bounds will be the same state because the environment froze
                bounds = self.returnState(copy.deepcopy(self._trueBoundaries) )
            else:
                done=False
                # The next state will be it's sibling
                bounds=self.StateBuffer[-1][0]
                bounds = self.returnState(copy.deepcopy(bounds) )
            # the reward is basically how off the action is,
            # we want for a zero area to output a termination
            # action[-2] is the prob of stop
            reward=(-1+action[-2])*5+self.penalty
            #print("< thresh reward ",reward,action[-2])
            return bounds, reward,done,None

        self.addStateAction( PrState, action )
        #print("state",PrState)
        #print("action",action)
        #self.render()
        #print("3val,PrState[dim][0],PrState[dim][1]",val,PrState[dim][0],PrState[dim][1])


        InfoGain,N1,N2=infoGain(df,dim,val,PrState[dim][0],PrState[dim][1],inheritedN)

        State1=copy.deepcopy(PrState)
        State2=copy.deepcopy(PrState)
        State1[dim][0]=val
        State2[dim][1]=val
        #print("state 1 and 2",State1,"\n",State2)
        # self.addState( State1, N1 )
        # self.addState( State2, N2 )
        reward = self._infoRewardMultiplier*(500*InfoGain+self.penalty)
        #print("cut reward",reward)
        #print("infogain",InfoGain)

        """
        np.random.randint(2, size=1)[0] gives either 0 or 1 with equal prob
        """
        if(np.random.randint(2, size=1)[0]==1):
            self.addState( State1, N1 )
            self.addState( State2, N2 )
            # bounds=self.returnState(copy.deepcopy(State2))
        else:
            self.addState( State2, N2 )
            self.addState( State1, N1 )
            # bounds=self.returnState(copy.deepcopy(State1))

        bounds=self.StateBuffer[-1][0]
        bounds = self.returnState(copy.deepcopy(bounds) )
        return bounds, reward,False,None
