import math
# This function basically boosts the amount of "Ether" samples
# in the off chance that they got diluted after
# successive cuts
def boostingInheritedN(df,inheritedN):
    Y=len(df.index)
    # We have handled the case in the previous env class,
    # where the df has zero elements, thus Y>0
    if(inheritedN<=Y):
        # This can be defined to any real number
        c=1
        inheritedN=c*Y

    return inheritedN


def info(df,inheritedN):
    #print("inherited N", inheritedN)
    Y=len(df.index)
    if(Y==0):
        ExpectedInfo = 0
        return ExpectedInfo

    probOfY=Y/(inheritedN+Y)
    probOfN=1-probOfY

    #print("printing Y,probOfY,inheritedN,Y, probOfN",Y,probOfY,inheritedN,Y,probOfN)
    infoY = -1*math.log(probOfY)
    infoN = -1*math.log(probOfN)
    ExpectedInfo = probOfY*infoY + probOfN*infoN
    # Note: information is always positive
    return ExpectedInfo

def infoAfterPartition(ModD1,D1Info,ModD2,D2Info):

    PartInfo = ( ( ModD1*D1Info + ModD2*D2Info )/ (ModD1+ModD2) )
    return PartInfo

def infoGain(df,dim,val,start,end,inheritedN):

    # we split our space based on the action taken

    df1=df.loc[df[str(dim)] >= val]
    df2=df.loc[df[str(dim)] < val]

    ModD1=len(df1.index)
    ModD2=len(df2.index)



    # This is based on the cluster Tree approach to update
    # the inheritedN
    inheritedN=boostingInheritedN(df,inheritedN)
    #print(end,start)
    N1=((end-val)*inheritedN)/(end-start)
    N2=((val-start)*inheritedN)/(end-start)
    #print("N1,N2",N1,N2)

    D1Info=info(df1,N1)
    D2Info=info(df2,N2)
    origInfo = info(df,inheritedN)
    PartInfo=infoAfterPartition(ModD1,D1Info,ModD2,D2Info)


    InfoGain = origInfo - PartInfo
    # Note: information is always positive, but delta info
    return InfoGain,N1,N2
