import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.matlib as matlib
import matplotlib.image as mpimg
import numpy as np
from numpy import zeros, newaxis
from scipy.io import loadmat


#this script extracts sample from 24 samples for TESTING. All 480,000 pixels for a sample is extracted which is the unseen data
for i in range(1,25):
    print('extracting sample {}'.format(i))
    wavelength = '550'
    samplepath = 'D:\EP dataset\Mueller\Sample{}\{}_Mueller.mat'.format(i,wavelength)
    textfile = open("C:/Users/krarv/Desktop/ProjectData/MaskedData/Sample{}Data.csv".format(i), "w")
    sample = loadmat(samplepath)

    data = sample.get('MM')

    cancermask = open('D:\EP dataset\Part_I\Sample{}\diseased_ZoneMask.dat'.format(i), 'rb')
    cancermask_data = np.fromfile(cancermask, dtype=np.uint32, count=600 * 800)
    cancermask_data = np.reshape(cancermask_data, [800, 600])
    cancermask_data = np.flip(cancermask_data, axis=1)
    cancermask_data = np.rot90(cancermask_data)

    healthymask = []
    if(i == 2):
        healthymask = open('D:\EP dataset\Part_I\Sample{}\healthy_ZoneMask_Original.dat'.format(i),'rb')
    else:
        healthymask = open('D:\EP dataset\Part_I\Sample{}\healthy_ZoneMask.dat'.format(i),'rb')
    healthymask_data = np.fromfile(healthymask, dtype=np.uint32, count=600 * 800)
    healthymask_data = np.reshape(healthymask_data, [800, 600])
    healthymask_data = np.flip(healthymask_data, axis=1)
    healthymask_data = np.rot90(healthymask_data)

    count = 1;
    for row in range(600):
        for col in range(800):
            if(healthymask_data[row][col] > 0.5):
                instance = []
                for dim in range(16):
                    input = data[row][col][dim]
                    instance.append(input)
                instance.append(0)
                instanceString = ','.join(str(i) for i in instance)
                textfile.write(instanceString + '\n')
                count = count + 1
            elif (cancermask_data[row][col] > 0.5):
                instance = []
                for dim in range(16):
                    input = data[row][col][dim]
                    instance.append(input)
                instance.append(1)
                instanceString = ','.join(str(i) for i in instance)
                textfile.write(instanceString + '\n')
                count = count + 1

    textfile.close()

