import json
import numpy as np
from anytree import AnyNode, RenderTree
from anytree.exporter import JsonExporter
from anytree.importer import JsonImporter
from anytree import LevelOrderGroupIter
from scipy.spatial import distance
import copy
from ExpertSlices import ExpertSlices
import pickle

def takeThird(elem):
    return elem[2]

class readTree:


    def __init__(self):
        self._trueBoundaries=np.array([[-1.,4.],[1.,6.]])
        self._maxRegressVal=2

        importer = JsonImporter()
        with open("../data/tree.txt") as json_file:
            data = json.load(json_file)

        root = importer.import_(data)


        for children in LevelOrderGroupIter(root):
            for node in children:
                # print(node.name)
                # arr=np.array(node.clusterPts)
                # mx=np.amax(arr, axis=0)
                # mn=np.amin(arr, axis=0)
                # mxmn=np.array([mn,mx])
                # node.clusterPts=mxmn

                """
                Note that bounds is a set
                Used to avoid duplication
                """
                node.bounds=set()
                # print(node.bounds)


                for child1 in node.children:
                    for child2 in node.children:
                        if(child1 is not child2):
                            #print(child1.name,child2.name)
                            distc1c2 = distance.cdist(child1.clusterPts,child2.clusterPts).min(axis=1)
                            distc2c1 = distance.cdist(child2.clusterPts,child1.clusterPts).min(axis=1)
                            minA=min(distc1c2)
                            minposA=np.argmin(distc1c2)
                            minposB=np.argmin(distc2c1)
                            # print(minposA,minposB,distance.cdist([child1.clusterPts[minposA]],[child2.clusterPts[minposB]]).min(axis=1))

                            # print("closest points",child1.clusterPts[minposA],child2.clusterPts[minposB])
                            midpt= (np.array(child1.clusterPts[minposA]) +  np.array(child2.clusterPts[minposB]) )/2
                            # print("midpoint",midpt)
                            all1Todim0=np.array(child1.clusterPts)[:,0]
                            all1Todim1=np.array(child1.clusterPts)[:,1]
                            all2Todim0=np.array(child2.clusterPts)[:,0]
                            all2Todim1=np.array(child2.clusterPts)[:,1]
                            if( all(midpt[0]<=i for i in  all1Todim0 ) or all(midpt[0]>=i for i in  all1Todim0 ) ):
                                if( all(midpt[0]<=i for i in  all2Todim0 ) or all(midpt[0]>=i for i in  all2Todim0 ) ):
                                    node.bounds.add(tuple([0,midpt[0],minA]))
                            else:
                                if( all(midpt[1]<=i for i in  all1Todim1 ) or all(midpt[1]>=i for i in  all1Todim1 ) ):
                                    if( all(midpt[1]<=i for i in  all2Todim1 ) or all(midpt[1]>=i for i in  all2Todim1 ) ):
                                        node.bounds.add(tuple([1,midpt[1],minA]))


                sortedList = sorted(node.bounds, key=takeThird)
                """
                We are going to take only the n-1 closest midpoints
                Why? Because each midpoint will become one partition
                and we only need n-1 partitions to partition n clusters
                """
                node.bounds=sortedList[:len(node.children)-1]

                #print(len(node.children))

        # for children in LevelOrderGroupIter(root):
        #     for node in children:
        #         node.perimeter=_trueBoundaries
        #         arr=np.array(node.clusterPts)
        #         mx=np.amax(arr, axis=0)
        #         mn=np.amin(arr, axis=0)
        #         mnmx=np.array([mn,mx]).transpose().tolist()
        #         node.clusterPts=mnmx
        # import statistics
        root.perimeter=copy.deepcopy(self._trueBoundaries)
        for children in LevelOrderGroupIter(root):
            for node in children:
                if(node is not root):
                    par=node.parent
                    node.perimeter=copy.deepcopy(par.perimeter)
                    # boundaries is of the form
                    # tuple([1,midpt[1],minA])
                    node.clusterPts=np.array(node.clusterPts)
                    maxEnclosedPts=0
                    for boundaries in par.bounds:


                        """
                        If the values of the midpoint (along the desired dimension)
                        are >= (or <=) all of the cluster points, then that partition
                        is a good partition
                        We are assuming xy separable
                        This commented code will only work for xy separable

                        if( ( boundaries[1]>= node.clusterPts[:,int(boundaries[0])] ).all() ):
                            if(node.perimeter[int(boundaries[0])][1]>=boundaries[1]):
                                node.perimeter[int(boundaries[0])][1]=boundaries[1]
                                #print(node.perimeter)
                        if( ( boundaries[1]<= node.clusterPts[:,int(boundaries[0])] ).all() ):
                            if(node.perimeter[int(boundaries[0])][0]<=boundaries[1]):
                                node.perimeter[int(boundaries[0])][0]=boundaries[1]
                                #print(node.perimeter)
                        """

                        """
                        We have two options for each cut
                        Either upperbound or lower bound
                        """
                        upb=sum( boundaries[1]>= node.clusterPts[:,int(boundaries[0])] )
                        lwb=sum( boundaries[1]<= node.clusterPts[:,int(boundaries[0])] )
                        if( upb>lwb ):
                            if(upb>maxEnclosedPts):
                                if(par.perimeter[int(boundaries[0])][1]>=boundaries[1]):
                                    maxEnclosedPts=upb
                                    node.perimeter[int(boundaries[0])][1]=boundaries[1]
                                    #print(node.perimeter)
                        else:
                            if(lwb>maxEnclosedPts):
                                if(par.perimeter[int(boundaries[0])][0]<=boundaries[1]):
                                    maxEnclosedPts=lwb
                                    node.perimeter[int(boundaries[0])][0]=boundaries[1]

                    #print("The final ",node.name,node.perimeter)
        slices=[]
        self.memory = ExpertSlices(int(1e4), 69)
        for children in LevelOrderGroupIter(root):
            for node in children:
                #print(node.name)
                # print(node.perimeter)
                # print(node.bounds)
                for slice in node.bounds:
                    #print("slice ",slice)
                    #print("peri ",node.perimeter)
                    actionMultiFactor=(self._trueBoundaries[int(slice[0])][1]-self._trueBoundaries[int(slice[0])][0])/self._maxRegressVal

                    val=(slice[1]-self._trueBoundaries[int(slice[0])][0])/actionMultiFactor
                    remainingPerimeter=np.delete(node.perimeter,int(slice[0]),0)
                    # print("final dim,val,variable vals",np.array([slice[0:2],remainingPerimeter ]) )

                    self.memory.add(slice[0],val,remainingPerimeter)
                    # slices.append( np.array([[slice[0],slice[1]],remainingPerimeter ])  )

    def act(self,state):
        # a=np.array([[2.,3.],[3.,4.]])
        ExpertAction=self.memory.action(state)
        return ExpertAction
