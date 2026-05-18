import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

def plot_check_filter(str):
    '''
     By the result of png, we know that the filter is: [[G R] [B G]]
    '''
    img = plt.imread(str)
    # chech the original filter
    sub_00 = img[0::2,0::2]
    sub_01 = img[0::2,1::2]
    sub_10 = img[1::2,0::2]
    sub_11 = img[1::2,1::2]

    subs = [sub_00, sub_01, sub_10, sub_11]

    # plot subplots for these 4 plots
    fig, ax = plt.subplots(2,2)
    for i in range(2):
        for j in range(2):
            ax[i,j].imshow(subs[i*2+j], cmap='gray')
    # save plots as png
    plt.savefig('original_filters.png')
    # show plots
    plt.show()


# Hamilton Adams to interpolate G
def Hamilton_adams(original_img):
    '''
    Args:
        original_img: original image
    Returns:
        interpolated green ndarray
    '''
    img_pad = np.pad(original_img,pad_width = 2,mode = 'reflect')
    R5 = img_pad[2:-2,2:-2] # red center

    # horizontal
    R3 = img_pad[2:-2,0:-4]
    G4 = img_pad[2:-2,1:-3]
    G6 = img_pad[2:-2,3:-1]
    R7 = img_pad[2:-2,4:]

    # verticle
    R1 = img_pad[0:-4,2:-2]
    G2 = img_pad[1:-3,2:-2]
    G8 = img_pad[3:-1,2:-2]
    R9 = img_pad[4:,2:-2]

    # edge detect
    dH = np.abs(G4-G6) + np.abs(2*R5-R3-R7)
    dV = np.abs(G2-G8) + np.abs(2*R5-R1-R9)


    # mask for green interpolation
    G5 = np.zeros_like(R5)
    mask_h = dH > dV
    G5[mask_h] = (G4[mask_h]+G6[mask_h])/2 + (2*R5[mask_h]-R3[mask_h]-R7[mask_h])/4

    mask_v = dH < dV
    G5[mask_v] = (G2[mask_v]+G8[mask_v])/2 + (2*R5[mask_v]-R1[mask_v]-R9[mask_v])/4

    mask_e = dH == dV
    G5[mask_e] = (G4[mask_e]+G6[mask_e]+G2[mask_e]+G8[mask_e])/4 + (4*R5[mask_e]-R1[mask_e]-R3[mask_e]-R7[mask_e]-R9[mask_e])/8

    return G5

# 色差補色
def color_diff(original_img,full_green,object_colormask,rest_colormask):
    '''
    Args:
        original_img: original image
        full_green: interpolated green ndarray
        object_colormask: the color already have to make diff value
        rest_colormask: for the other color that need to be calculated by shifting
    Returns:
        single color channel
    '''
    kr = np.zeros_like(full_green,dtype=np.float32)
    # 1. color we have
    kr[object_colormask] = original_img[object_colormask] - full_green[object_colormask]

    # 2. color we do not have
    k_pad = np.pad(kr,pad_width=1,mode="reflect")
    k_ul = k_pad[0:-2,0:-2]
    k_ur = k_pad[0:-2,2:]
    k_dl = k_pad[2:,0:-2]
    k_dr = k_pad[2:,2:]

    kr[rest_colormask] = (k_ul[rest_colormask]+k_ur[rest_colormask]+k_dl[rest_colormask]+k_dr[rest_colormask])/4

    # 3. green point to calculate k with 上下左右差值
    k_pad = np.pad(kr,pad_width=1,mode="reflect")
    k_u = k_pad[0:-2,1:-1]
    k_d = k_pad[2:,1:-1]
    k_l = k_pad[1:-1,0:-2]
    k_r = k_pad[1:-1,2:]
    green_mask = ~(object_colormask|rest_colormask)
    kr[green_mask] = (k_d[green_mask]+k_u[green_mask]+k_l[green_mask]+k_r[green_mask])/4

    # median filter to smooth
    kr = median_filter(kr,size=3)

    full_color_channel = full_green + kr
    return full_color_channel

img = plt.imread('for_demosaic.png')
full_green = img.copy().astype(np.float32)
interpolation_green = Hamilton_adams(img)

# By the result of png, we know that the filter is: [[G R] [B G]]
is_red = np.zeros_like(img,dtype=bool)
is_blue = np.zeros_like(img,dtype= bool)
is_red[0::2,1::2] = True
is_blue[1::2,0::2] = True

interpolation_green_mask = is_blue|is_red
full_green[interpolation_green_mask] = interpolation_green[interpolation_green_mask]

full_red = color_diff(img,full_green,is_red,is_blue)
full_blue = color_diff(img,full_green,is_blue,is_red)

# combine three channel together BGR
# stack -> clip -> unit8
color_img = np.stack([full_red,full_green,full_blue],axis=2)
if img.max()<=1.0:
    color_img = color_img*255
color_img = np.clip(color_img,0,255)
color_img = color_img.astype(np.uint8)

# save the final image
plt.imsave('color_img.png', color_img)
