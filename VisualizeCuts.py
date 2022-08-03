import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.matlib as matlib
import matplotlib.image as mpimg
import numpy as np
from numpy import zeros,newaxis

from scipy.io import loadmat
axes = []
fig = plt.figure()
for a in range (1,25):
    wavelength = '600'
    samplepath = 'D:/EP dataset/Part_I/Sample{}/{}_Intensity.mat'.format(a,wavelength)
    sample = loadmat(samplepath)
    mask = loadmat('D:/EP dataset/Part_I/Sample{}/Sample{}_Mask.mat'.format(a,a))
    data = sample.get('IN');
    ColorMask = mask.get('ColorMask')
    N = data[:,:,1]
    x_values = ColorMask[:,:,0] * ColorMask[:,:,1] * ColorMask[:,:,2]
    print(x_values.shape)
    mask_threshold = 1
    y_values = np.ma.array(data[:,:,0])
    y_values_masked = np.ma.masked_where(x_values > mask_threshold, y_values)

    axes.append(fig.add_subplot(5, 5, a))
    subplot_title = "Sample" + str(a)
    axes[-1].set_title(subplot_title)
    plt.imshow(y_values_masked)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.tight_layout()
plt.show()