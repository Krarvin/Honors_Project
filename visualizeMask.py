import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import zeros,newaxis
from matplotlib import cm
from PIL import Image
from skimage.io import imread

from scipy.io import loadmat
wavelength = '600'
mask_threshold = 1
samplepath = 'D:\EP dataset\Part_I\Sample1\{}_Intensity.mat'.format(wavelength)
sample = loadmat(samplepath)
cancermask = open('D:\EP dataset\Part_I\Sample1\diseased_ZoneMask.dat', 'rb')
healthymask = open('D:\EP dataset\Part_I\Sample1\healthy_ZoneMask.dat', 'rb')

cancermask_data = np.fromfile(cancermask,dtype=np.uint32, count=600*800)
#reshape mask so it has correct position relative to cervical cancer cell
reshape = np.reshape(cancermask_data,[800,600])
reshape = np.flip(reshape, axis = 1)
cancer_reshape = np.rot90(reshape)
#plt.imshow(cancer_reshape, cmap='Reds')

healthy_data = np.fromfile(healthymask,dtype=np.uint32, count=600*800)
reshape = np.reshape(healthy_data,[800,600])
reshape = np.flip(reshape, axis = 1)
healthy_reshape = np.rot90(reshape)
#plt.imshow(healthy_reshape, cmap='Greens')

sample_data = sample.get('IN')
x_values = cancer_reshape
y_values = np.ma.array(sample_data[:,:,1])
y_values_masked = np.ma.masked_where(x_values>mask_threshold, y_values)

cmap = matplotlib.cm.gray
cmap.set_bad(color='red')
plt.imshow(y_values_masked, cmap =cmap,interpolation='nearest')

x_values = healthy_reshape
y_values = np.ma.masked_where(x_values<mask_threshold, y_values)
plt.imshow(y_values, cmap = 'Greens', alpha = 25)
plt.show()
