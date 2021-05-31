# -*- coding: utf-8 -*-


import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter
import argparse
import  skfmm
import imageio
import matplotlib.pyplot as plt


## signed geodesic distance
def phi(mask):

    phi_ext = skfmm.distance(np.max(mask)-mask)
    phi_int = skfmm.distance(mask)

    return phi_ext - phi_int

## signed Euclidean distance
def phi_Euclidean(mask):

    phi_ext = ndimage.distance_transform_edt(np.max(mask)-mask)
    phi_int = ndimage.distance_transform_edt(mask)

    return phi_ext - phi_int



def curvature(phi):

    g_x,g_y = np.gradient(phi)
    #smoothing of gradient vector field
    gaussian_filter(g_x, sigma=2, output=g_x)
    gaussian_filter(g_y, sigma=2, output=g_y)
    norm_grad =  np.sqrt(np.power(g_x,2)+np.power(g_y,2))
    norm_grad[np.where(norm_grad==0)]=1
    np.divide(g_x,norm_grad,g_x)
    np.divide(g_y,norm_grad,g_y)
    g_xx, g_yx  = np.gradient(g_x)
    g_xy, g_yy  = np.gradient(g_y)
    gaussian_filter(g_xx, sigma=2, output=g_xx)
    gaussian_filter(g_yy, sigma=2, output=g_yy)

    return 0.5*(g_xx + g_yy)

def plot_curvature(phi, curvature, image, out_name):

    contours = np.where(np.logical_and(phi<=0.4, phi>=-0.4))
    x,y = contours[0], contours[1]
    plt.imshow(image,cmap='gray')
    plt.scatter(y, x, s=1, c= curvature[x,y], cmap='jet')
    plt.axis("equal")
    plt.colorbar(shrink=0.95)
    plt.clim(np.min(curvature[x,y]),np.max(curvature[x,y]))
    plt.axis('off')
    plt.margins(0,0)
    plt.rcParams['figure.facecolor'] = 'black'
    plt.savefig(out_name, dpi=500)
    plt.show()

    return 0



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--mask', help='2D shape binary mask, as png file', type=str, required = True)
    parser.add_argument('-anat', '--anatomical', help='anatomical image, as png file', type=str, required = True)
    parser.add_argument('-o', '--output', help='output directory', type=str, default = './curvature_results')

    args = parser.parse_args()

    # Example of use : python3 fast_mean_curvature_2D.py -anat ./2D_data/anatomical_T1.png -in ./2D_data/wm.png

    output_path = args.output

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    shape = imageio.imread(args.mask)

    anat = imageio.imread(args.anatomical)


    phi = phi(shape)

    gaussian_filter(phi, sigma=2, output=phi)

    curvature = curvature(phi)

    plot_curvature(phi, curvature, anat, output_path+'/mean_curvature.png')
