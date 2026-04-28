import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('for_demosaic.png')

# chech the original filter
# sub_00 = img[0::2,0::2]
# sub_01 = img[0::2,1::2]
# sub_10 = img[1::2,0::2]
# sub_11 = img[1::2,1::2]

# subs = [sub_00, sub_01, sub_10, sub_11]

# # plot subplots for these 4 plots
# fig, ax = plt.subplots(2,2)
# for i in range(2):
#     for j in range(2):
#         ax[i,j].imshow(subs[i*2+j], cmap='gray')
# # save plots as png
# plt.savefig('original_filters.png')
# # show plots
# plt.show()

# by the result of png, we know that the filter is: [[G R] [B G]]