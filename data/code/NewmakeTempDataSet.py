import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

from anytree import AnyNode, RenderTree
from anytree.exporter import JsonExporter
from anytree.importer import JsonImporter
import json

import numpy as np
from scipy.spatial import distance

N = 20
N2= 20
g1 = 1+np.random.rand(N)
g2 = -1+np.random.rand(N)
g3 = 3+np.random.rand(N)
# new data points
g21 = 0.5+np.random.rand(N2)
g31 = 1.5+np.random.rand(N2)

g4 = 3 + np.random.rand(N)
g5 = 5 + np.random.rand(N)
g6 = 3 + np.random.rand(N)

# new data points
g41 = 5+np.random.rand(N2)
g51 = 1+np.random.rand(N2)

x=np.append(g1,g2)
x=np.append(x,g3)
# new appended x
x=np.append(x,g21)
x=np.append(x,g31)


y=np.append(g4,g5)
y=np.append(y,g6)
# new appended y
y=np.append(y,g41)
y=np.append(y,g51)

data=np.array([x,y])
data=data.T
np.savetxt("../NewdataPts.csv", data, delimiter=",")
#print(data.shape)

df=pd.DataFrame(data)

# plt.scatter(df[0], df[1])
# plt.title('Scatter plot')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

s2Pts=[]
s00Pts=[]
s01Pts=[]
s10Pts=[]
s11Pts=[]
temp=[]
for i in range(5):
    #print(df[0][N*i],df[1][N*i])
    for j in range(N):
        temp.append([df[0][N*i+j],df[1][N*i+j] ])
    if(i==0):
        s10Pts=copy.deepcopy(temp)
    if(i==1):
        s00Pts=copy.deepcopy(temp)
    if(i==2):
        s11Pts=copy.deepcopy(temp)
    if(i==3):
        s01Pts=copy.deepcopy(temp)
    if(i==4):
        s2Pts=copy.deepcopy(temp)
    temp=[]

s0Pts=s00Pts+s01Pts
s1Pts=s10Pts+s11Pts
rootPts=s0Pts+s1Pts+s2Pts
#print(len(rootPts))



root = AnyNode(name="Root", clusterPts=rootPts)
s0 = AnyNode(name="sub0", parent=root,clusterPts=s0Pts)
s0a = AnyNode(name="sub00", parent=s0,clusterPts=s00Pts)
s0b = AnyNode(name="sub01", parent=s0,clusterPts=s01Pts)
s1 = AnyNode(name="sub1", parent=root,clusterPts=s1Pts)
s1a = AnyNode(name="sub10", parent=s1,clusterPts=s10Pts)
s1b = AnyNode(name="sub11", parent=s1,clusterPts=s11Pts)
s2 = AnyNode(name="sub2", parent=root,clusterPts=s2Pts)

#print(RenderTree(root))
for pre, fill, node in RenderTree(root):
    print("%s%s" % (pre, node.name))

# For exporting
exporter = JsonExporter(indent=2, sort_keys=True)
print(exporter.export(root))

with open('tree.txt', 'w') as outfile:
    json.dump(exporter.export(root), outfile)


##### How to find the midpoints of two lists
minA=min(distance.cdist(s1.clusterPts,s2.clusterPts).min(axis=1))
print(minA)
minposA=np.argmin(distance.cdist(s1.clusterPts,s2.clusterPts).min(axis=1))
minposB=np.argmin(distance.cdist(s2.clusterPts,s1.clusterPts).min(axis=1))
print(minposA,minposB,distance.cdist([s1.clusterPts[minposA]],[s2.clusterPts[minposB]]).min(axis=1))

print("closest points",s1.clusterPts[minposA],s2.clusterPts[minposB])
midpt= (np.array(s1.clusterPts[minposA]) +  np.array(s2.clusterPts[minposB]) )/2
print("midpoint",midpt)
all2dim0=np.array(s1.clusterPts)[:,0]
all2dim1=np.array(s1.clusterPts)[:,1]
print(  all(midpt[0]<=i for i in  all2dim0 ) or all(midpt[0]>=i for i in  all2dim0 ) )
print(  all(midpt[1]<=i for i in  all2dim1 ) or all(midpt[1]>=i for i in  all2dim1 ) )
