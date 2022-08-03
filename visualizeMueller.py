import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math


from scipy.io import loadmat
wavelength = '600'
mask_threshold = 1
sample_number = 1
samplepath = 'D:\EP dataset\Mueller\Sample{}\{}_Mueller.mat'.format(sample_number,wavelength)
sample = loadmat(samplepath)
cancermask = open('D:\EP dataset\Part_I\Sample{}\healthy_ZoneMask.dat'.format(sample_number), 'rb')

data = sample.get('MM');
cancermask_data = np.fromfile(cancermask,dtype=np.uint32, count=600*800)
cancermask_data = np.reshape(cancermask_data,[800,600])
cancermask_data = np.flip(cancermask_data, axis = 1)
cancermask_data = np.rot90(cancermask_data)

x_values = cancermask_data
y_values = np.ma.array(data[:, :, 0])
mask = np.ma.masked_where(x_values < mask_threshold, y_values)

plt.imshow(mask)
plt.show()

axes = []
fig = plt.figure()
count = 1
count2 = 1
for i in range(16):
    axes.append(fig.add_subplot(4, 4, i + 1))
    subplot_title = "m" + str(count) + ":" + str(count2)
    if(count2 == 4):
        count2 = 1
        count = count + 1
    else:
        count2 = count2 + 1
    axes[-1].set_title(subplot_title)
    plt.imshow(data[:,:,i])
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.tight_layout()
plt.show()