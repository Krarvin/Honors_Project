import numpy as np
from scipy.io import loadmat


#this script extracts masked data from 24 samples for TRAINING

wavelength = '550'
mask_threshold = 1
textfile = open("C:/Users/krarv/Desktop/project16Data.csv", "w")
contains_healthy = [1,2,4,5,7,8,9,11,12,13,14,18,19,20,22]
contains_cancer = [3,6,10,15,16,17,18,19,21,22,23,24]

for a in contains_cancer:
    samplepath = 'D:\EP dataset\Mueller\Sample{}\{}_Mueller.mat'.format(str(a), wavelength)
    sample = loadmat(samplepath)
    cancermask = open('D:\EP dataset\Part_I\Sample{}\diseased_ZoneMask.dat'.format(str(a)), 'rb')

    sample_data = sample.get('MM')
    cancermask_data = np.fromfile(cancermask, dtype=np.uint32, count=600 * 800)
    cancermask_data = np.reshape(cancermask_data, [800, 600])
    cancermask_data = np.flip(cancermask_data, axis=1)
    cancermask_data = np.rot90(cancermask_data)
    print('extracting sample {}'.format(str(a)))


    x_values = cancermask_data
    y_values = np.ma.array(sample_data[:, :, 0])
    mask = np.ma.masked_where(x_values < mask_threshold, y_values)

    masked_dimensions = []
    for i in range(16): #mask all dimensions of Mueller Matrix
        y_values = np.ma.array(sample_data[:, :, i])
        masked = np.ma.masked_where(x_values < mask_threshold, y_values)
        masked_dimensions.append(masked)

    count = 1;

    for row in range(600):
        for col in range(800):
            if (mask[row][col] > 0):
                instance = []
                for dim in range(16):
                    instance.append(sample_data[row][col][dim]) #extract pixel to CSV
                instance.append(1)
                instanceString = ','.join(str(i) for i in instance)
                textfile.write(instanceString + '\n')
            count = count + 1

for a in contains_healthy:
    samplepath = 'D:\EP dataset\Mueller\Sample{}\{}_Mueller.mat'.format(str(a), wavelength)
    sample = loadmat(samplepath)
    if(a == 2):
        healthymask = open('D:\EP dataset\Part_I\Sample{}\healthy_ZoneMask_Original.dat'.format(str(a)), 'rb')
    else:
        healthymask = open('D:\EP dataset\Part_I\Sample{}\healthy_ZoneMask.dat'.format(str(a)), 'rb')
    sample_data = sample.get('MM')
    healthymask_data = np.fromfile(healthymask, dtype=np.uint32, count=600 * 800)
    healthymask_data = np.reshape(healthymask_data, [800, 600])
    healthymask_data = np.flip(healthymask_data, axis=1)
    healthymask_data = np.rot90(healthymask_data)
    print('extracting sample {}'.format(str(a)))

    x_values = healthymask_data
    y_values = np.ma.array(sample_data[:, :, 0])
    mask = np.ma.masked_where(x_values < mask_threshold, y_values)

    masked_dimensions = []
    for i in range(16):
        y_values = np.ma.array(sample_data[:, :, i])
        masked = np.ma.masked_where(x_values < mask_threshold, y_values)
        masked_dimensions.append(masked)

    count = 1
    for row in range(600):
        for col in range(800):
            #print(str(count) + ' healthy')
            if (mask[row][col] > 0):
                instance = []
                for dim in range(16):
                    instance.append(sample_data[row][col][dim])
                instance.append(0)
                instanceString = ','.join(str(i) for i in instance)
                textfile.write(instanceString + '\n')
            count = count + 1

textfile.close()
