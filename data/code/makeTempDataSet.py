import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

from anytree import AnyNode, RenderTree
from anytree.exporter import JsonExporter
# import graphviz


N = 20
g1 = 1+np.random.rand(N)
g2 = -1+np.random.rand(N)
g3 = 3+np.random.rand(N)
g4 = 3 + np.random.rand(N)
g5 = 5 + np.random.rand(N)
g6 = 3 + np.random.rand(N)
x=np.append(g1,g2)
x=np.append(x,g3)

y=np.append(g4,g5)
y=np.append(y,g6)
data=np.array([x,y])
data=data.T
np.savetxt("../dataPts.csv", data, delimiter=",")
print(data.shape)

df=pd.DataFrame(data)

plt.scatter(df[0], df[1])
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



s0Pts=[]
s10Pts=[]
s11Pts=[]
temp=[]
for i in range(3):
    for j in range(N):
        temp.append([df[0][N*i+j],df[1][N*i+j] ])
    if(i==0):
        s10Pts=copy.deepcopy(temp)
    if(i==1):
        s0Pts=copy.deepcopy(temp)
    if(i==2):
        s11Pts=copy.deepcopy(temp)
    temp=[]

s1Pts=s11Pts+s10Pts
rootPts=s1Pts+s0Pts
print(len(rootPts))



root = AnyNode(name="Root", clusterPts=rootPts)
s0 = AnyNode(name="sub0", parent=root,clusterPts=s0Pts)
s1 = AnyNode(name="sub1", parent=root,clusterPts=s1Pts)
s1a = AnyNode(name="sub10", parent=s1,clusterPts=s10Pts)
s1b = AnyNode(name="sub11", parent=s1,clusterPts=s11Pts)

#print(RenderTree(root))
for pre, fill, node in RenderTree(root):
    print("%s%s" % (pre, node.name))
