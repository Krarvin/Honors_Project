import matplotlib
from scipy.io import loadmat
from keras.models import model_from_json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

#this script predicts an entire sample of unseen data using a pretrained model

wavelength = '550'
sample_number = 10
loaded_model = pickle.load(open('mlp_model.sav','rb')) #use to load Decison Tree/MLP model
data = pd.read_csv("C:/Users/krarv/Desktop/ProjectData/SampleData/Sample{}Data.csv".format(sample_number),header=None,delimiter=',')
x = data
predictions = loaded_model.predict(x)
samplepath = 'D:/EP dataset/Part_I/Sample{}/{}_Intensity.mat'.format(sample_number,wavelength)
sample = loadmat(samplepath)
intensity_data = sample.get('IN')


cancermask = open('D:\EP dataset\Part_I\Sample{}\diseased_ZoneMask.dat'.format(sample_number),'rb')
cancermask_data = np.fromfile(cancermask, dtype=np.uint32, count=600 * 800)
cancermask_data = np.reshape(cancermask_data, [800, 600])
cancermask_data = np.flip(cancermask_data, axis=1)
cancermask_data = np.rot90(cancermask_data)

healthymask = open('D:\EP dataset\Part_I\Sample{}\healthy_ZoneMask.dat'.format(sample_number),'rb')
healthymask_data = np.fromfile(healthymask, dtype=np.uint32, count=600 * 800)
healthymask_data = np.reshape(healthymask_data, [800, 600])
healthymask_data = np.flip(healthymask_data, axis=1)
healthymask_data = np.rot90(healthymask_data)


maskImage = [] #EP masks to compare to
for row in range(600):
    instance = []
    for col in range(800):
        if(healthymask_data[row][col] > 0.5):
            instance.append((0,255,0))
        elif(cancermask_data[row][col] > 0.5):
            instance.append((255,0,0))
        else:
            instance.append((255, 255, 255))
    maskImage.append(instance)

count = 0
output_matrix = [] #predicted mask values to compare to
for row in range(600): #append predictions to 600x800 matrix
    instance = []
    for col in range (800):
        instance.append(predictions[count])
        count = count + 1
    output_matrix.append(instance)
output_matrix = np.array(output_matrix)
mask = cancermask_data + healthymask_data
mask = np.array(mask)
thresholdImage = []
for row in range(600): #converts binary values (0,1) to green and red
    instance = []
    for col in range(800):
        if(output_matrix[row][col] > 0.5 and mask[row][col] > 0.5):
            instance.append((255,0,0))
        elif(output_matrix[row][col] < 0.5 and mask[row][col] > 0.5):
            instance.append((0,255,0))
        else:
            instance.append((255,255,255))
    thresholdImage.append(instance)


predictedImage = [] #predicts the entire image of unseen test data
for row in range(600):
    instance = []
    for col in range(800):
        if(output_matrix[row][col] > 0.5):
            instance.append((255,0,0))
        elif(output_matrix[row][col] < 0.5):
            instance.append((0,255,0))
        else:
            instance.append((255,255,255))
    predictedImage.append(instance)

mask = loadmat('D:/EP dataset/Part_I/Sample{}/Sample{}_Mask.mat'.format(sample_number,sample_number))
mask = mask.get('ColorMask')
for row in range(600):#apply mask of histopathological cuts to predicted image
    for col in range (800):
        for rgb in range (3):
            if(mask[row][col][rgb]> 0):
                predictedImage[row][col] = mask[row][col]


intensity_image = intensity_data[:,:,0]
plt.imshow(predictedImage)
plt.show()

mask_threshold = 1
x_values = cancermask_data
y_values = np.ma.array(intensity_image)
intensity_image = np.ma.masked_where(x_values>mask_threshold, y_values)

for row in range(600):
    for col in range (800):
        for rgb in range (3):
            if(mask[row][col][rgb]> 0):
                intensity_image[row][col] = 40000

plt.imshow(predictedImage)
plt.show()
plt.subplot(2,2,1),plt.imshow(maskImage)
plt.title('Sample {} - EP Prediction'.format(sample_number)), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(thresholdImage)
plt.title('Sample {} - Neural Network Prediction - Masked'.format((sample_number))), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(predictedImage)
plt.title('Sample {} - Neural Network Prediction - Full Sample'.format(sample_number)), plt.xticks([]), plt.yticks([])
cmap = matplotlib.cm.gray
cmap.set_bad(color='red')
plt.subplot(2,2,4),plt.imshow(intensity_image, cmap =cmap,interpolation='nearest')
x_values = healthymask_data
intensity_image = np.ma.masked_where(x_values<mask_threshold, y_values)
plt.imshow(intensity_image, cmap = 'Greens', alpha = 25)
plt.title('Sample {} - Intensity Image(wavelength={},M1:1)'.format(sample_number,wavelength)), plt.xticks([]), plt.yticks([])
plt.show()