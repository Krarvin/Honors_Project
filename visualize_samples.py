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


axes = []
fig = plt.figure()
for a in range (24):
    samplepath = 'D:\EP dataset\Intensity_Data\Sample{}\{}_Intensity.mat'.format(str(a+1),wavelength)
    sample = loadmat(samplepath)
    cancermask = open('D:\EP dataset\Intensity_Data\Sample{}\diseased_ZoneMask.dat'.format(str(a+1)), 'rb')
    healthymask = open('D:\EP dataset\Intensity_Data\Sample{}\healthy_ZoneMask.dat'.format(str(a+1)), 'rb')
    cancermask_data = np.fromfile(cancermask, dtype=np.uint32, count=600 * 800)
    # reshape mask so it has correct position relative to cervical cancer cell
    reshape = np.reshape(cancermask_data, [800, 600])
    reshape = np.flip(reshape, axis=1)
    cancer_reshape = np.rot90(reshape)

    healthy_data = np.fromfile(healthymask, dtype=np.uint32, count=600 * 800)
    reshape = np.reshape(healthy_data, [800, 600])
    reshape = np.flip(reshape, axis=1)
    healthy_reshape = np.rot90(reshape)

    sample_data = sample.get('IN')
    x_values = cancer_reshape
    y_values = np.ma.array(sample_data[:, :, 1])
    y_values_masked = np.ma.masked_where(x_values > mask_threshold, y_values)


    axes.append(fig.add_subplot(5,5,a+1))
    subplot_title = "Sample" + str(a + 1)
    axes[-1].set_title(subplot_title)
    cmap = matplotlib.cm.gray
    cmap.set_bad(color='red')
    plt.imshow(y_values_masked, cmap=cmap)

    x_values = healthy_reshape
    y_values = np.ma.masked_where(x_values < mask_threshold, y_values)
    plt.imshow(y_values, cmap='Greens', alpha=25)

    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
plt.show()
