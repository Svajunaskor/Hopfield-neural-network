from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def readimagetoarray(file):
    imagein = Image.open(file).convert(mode="L")
    imagearray = np.asarray(imagein,dtype=np.uint8)
    data = np.zeros(imagearray.shape,dtype=np.float)
    data[imagearray > 0] = -1
    data[imagearray == 0] = 1
    data =np.reshape(data,-1)
    return(data)
    
    
def arraytoimage(xinp):
    xarray =np.zeros((12,10))
    n = 0
    for t in range(12):
        for y in range(10):
            xarray[t][y]=xinp[n]
            n+= 1
    plt.matshow(xarray, cmap='Greys')
    return()

def weightmatrix(data):
    lengthofdata=len(data)
    identitymatrix = np.identity(lengthofdata)
    weightm=np.zeros((lengthofdata,lengthofdata))
    for i in range(lengthofdata):
        for j in range(lengthofdata):
            weightm[i][j]=data[i]*data[j]       
    weightm-=identitymatrix
    return(weightm)      

def learn(learndata):
    Mweight=np.zeros((len(learndata[0]),len(learndata[0])))
    for i in range(len(learndata)):
        temp = weightmatrix(learndata[i])
        Mweight+=temp  
    return (Mweight)


def liapunov(Mweight, xdata):
    E = 0
    for i in range(len(xdata)):
        for j in range(len(xdata)):    
            E+=-(1/2)*Mweight[i][j]*xdata[i]*xdata[j]
    return E


def updaterandomneurons(Mweight,xdata):
    updateddata = xdata
    randarray = np.random.randint(len(xdata), size=60)
    for k in range(len(randarray)):
        tempvalue = 0
        for l in range(len(xdata)):
            tempvalue +=Mweight[randarray[k]][l]*xdata[l]
        if tempvalue > -1:
            updateddata[randarray[k]]=1
        if tempvalue < -1:
            updateddata[randarray[k]]=-1
    xdata=updateddata
    return(xdata)


def testrecognition(testdata, Mweight):
    for k in range(len(testdata)):
        xdata=testdata[k]
        Et=0
        E=2
        while not Et == E:
            E=Et
            arraytoimage(xdata)
            xdata = updaterandomneurons(Mweight,xdata)
            Et = liapunov(Mweight, xdata)
    return ()

#main function
def main():
    learnfiles=['one.png','A.png', 'X.png', 'Y.png', 'Z.png' , 'C.png' ]
    testfiles=['Atest.png']
    learndata=[]
    testdata=[]
    for i in range(len(learnfiles)):
        learndata.append(readimagetoarray(learnfiles[i]))
        arraytoimage(learndata[i])
    for j in range(len(testfiles)):    
        testdata.append(readimagetoarray(testfiles[j]))
        arraytoimage(testdata[j])
    Mweight= learn(learndata)
    testrecognition(testdata, Mweight)
    return()

main()
    