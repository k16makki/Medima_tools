# -*- coding: utf-8 -*-


import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter
import argparse
import  skfmm
import imageio
import matplotlib.pyplot as plt


import fast_Gaussian_curvature_3D as g3D




def L2_norm_grad2D(gx,gy):

    norm_grad =  np.sqrt(np.square(gx)+np.square(gy))
    gaussian_filter(norm_grad, sigma=1, output=norm_grad)
    norm_grad[np.where(norm_grad==0)]=1 # just to avoid dividing by zero

    return  norm_grad



def hessian_trace2D(hessian):

    return hessian[0,0,...] + hessian[1,1,...]


def mean_curvature(phi_grad,hessian):

    gx, gy = phi_grad

    mean_curv =  (gx * (gx*hessian[0,0,...]+gy*hessian[1,0,...]) + gy * (gx*hessian[0,1,...]+gy*hessian[1,1,...]))\
    - (L2_norm_grad2D(gx,gy)**2 *  hessian_trace2D(hessian))

    np.divide(mean_curv,-2*L2_norm_grad2D(gx,gy)**3,mean_curv)
    gaussian_filter(mean_curv, sigma=2, output=mean_curv)

    return mean_curv


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

    contours = np.where(np.logical_and(phi<=0.5, phi>=-0.5))
    x,y = contours[0], contours[1]
    plt.imshow(image,cmap='gray',origin='lower')
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

    # Example of use : python3 fast_mean_curvature_2D.py -anat /home/karim/Bureau/Courbure/2D_data/anatomical_T1.png
    #-in /home/karim/Bureau/Courbure/2D_data/cortex.png -o /home/karim/Bureau/Courbure/test2D

    output_path = args.output

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    shape = imageio.imread(args.mask)

    anat = imageio.imread(args.anatomical)


    phi = g3D.phi(shape)

    gaussian_filter(phi, sigma=2, output=phi)

    ########## Compute mean curvature ###################

    #curvature = curvature(phi)

    phi_grad, hessian = g3D.hessian(phi)
    curvature = mean_curvature(phi_grad,hessian)

    ######################################################


    plot_curvature(phi, curvature, anat, output_path+'/mean_curvature.png')
