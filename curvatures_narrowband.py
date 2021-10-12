# -*- coding: utf-8 -*-

"""
  ©
  Author: Karim Makki
"""

import trimesh
import numpy as np
import nibabel as nib
import os
from scipy.ndimage.filters import gaussian_filter
import argparse
import  skfmm
from skimage import measure
import timeit
import fast_Gaussian_curvature_3D as g3D


### Distance calculation limited to narrow band

def phi_narrow(mask, band=5):

    tmp = np.ones(mask.shape)
    tmp[mask!=0]= -1
    sgd = np.array(skfmm.distance(tmp, narrow=band), float)
    R = np.where(sgd != 0)
    sgd[sgd == 0] = 1

    return  sgd, R

def local_gaussian_filter(scalar_function, sigma=2):

    mask = np.zeros(scalar_function.shape)
    mask[scalar_function!=0] = 1
    smooth_scalar_function = gaussian_filter(scalar_function*mask, sigma=sigma)

    return smooth_scalar_function

def hessian_adjoint_narrowband(hessian,R):

    Ha = np.zeros(hessian.shape)
    Ha[0,0,R[0],R[1],R[2]] = hessian[1,1,R[0],R[1],R[2]]*hessian[2,2,R[0],R[1],R[2]] - hessian[1,2,R[0],R[1],R[2]]*hessian[2,1,R[0],R[1],R[2]]
    Ha[0,1,R[0],R[1],R[2]] = hessian[1,2,R[0],R[1],R[2]]*hessian[2,0,R[0],R[1],R[2]] - hessian[1,0,R[0],R[1],R[2]]*hessian[2,2,R[0],R[1],R[2]]
    Ha[0,2,R[0],R[1],R[2]] = hessian[1,0,R[0],R[1],R[2]]*hessian[2,1,R[0],R[1],R[2]] - hessian[1,1,R[0],R[1],R[2]]*hessian[2,0,R[0],R[1],R[2]]

    Ha[1,0,R[0],R[1],R[2]] = hessian[0,2,R[0],R[1],R[2]]*hessian[2,1,R[0],R[1],R[2]] - hessian[0,1,R[0],R[1],R[2]]*hessian[2,2,R[0],R[1],R[2]]
    Ha[1,1,R[0],R[1],R[2]] = hessian[0,0,R[0],R[1],R[2]]*hessian[2,2,R[0],R[1],R[2]] - hessian[0,2,R[0],R[1],R[2]]*hessian[2,0,R[0],R[1],R[2]]
    Ha[1,2,R[0],R[1],R[2]] = hessian[0,1,R[0],R[1],R[2]]*hessian[2,0,R[0],R[1],R[2]] - hessian[0,0,R[0],R[1],R[2]]*hessian[2,1,R[0],R[1],R[2]]

    Ha[2,0,R[0],R[1],R[2]] = hessian[0,1,R[0],R[1],R[2]]*hessian[1,2,R[0],R[1],R[2]] - hessian[0,2,R[0],R[1],R[2]]*hessian[1,1,R[0],R[1],R[2]]
    Ha[2,1,R[0],R[1],R[2]] = hessian[1,0,R[0],R[1],R[2]]*hessian[0,2,R[0],R[1],R[2]] - hessian[0,0,R[0],R[1],R[2]]*hessian[1,2,R[0],R[1],R[2]]
    Ha[2,2,R[0],R[1],R[2]] = hessian[0,0,R[0],R[1],R[2]]*hessian[1,1,R[0],R[1],R[2]] - hessian[0,1,R[0],R[1],R[2]]*hessian[1,0,R[0],R[1],R[2]]

    return Ha


def L2_norm_grad_narrowband(gx,gy,gz,R):

    norm_grad = np.zeros(gx.shape)
    norm_grad[R] =  np.sqrt(gx[R]**2 + gy[R]**2 + gz[R]**2)
    norm_grad =  local_gaussian_filter(norm_grad, sigma=1)
    norm_grad[np.where(norm_grad==0)]=1 # just to avoid dividing by zero

    return  norm_grad

def hessian_trace_narrowband(hessian,R):

    return hessian[0,0,R[0],R[1],R[2]] + hessian[1,1,R[0],R[1],R[2]] + hessian[2,2,R[0],R[1],R[2]]


def curvatures_narrowband(phi_grad,Ha,hessian,R):

    gx, gy, gz = phi_grad
    norm = L2_norm_grad_narrowband(gx,gy,gz,R)
    gx /= norm
    gy /= norm
    gz /= norm

    gaussian_curv = np.zeros(gx.shape)
    mean_curv = np.zeros(gx.shape)

    gaussian_curv[R] =  gx[R] * (gx[R]*Ha[0,0,R[0],R[1],R[2]]+gy[R]*Ha[1,0,R[0],R[1],R[2]]+\
    gz[R]*Ha[2,0,R[0],R[1],R[2]]) + gy[R] *(gx[R]*Ha[0,1,R[0],R[1],R[2]]+gy[R]*\
    Ha[1,1,R[0],R[1],R[2]]+gz[R]*Ha[2,1,R[0],R[1],R[2]])+ gz[R] * (gx[R]*Ha[0,2,R[0],R[1],R[2]]\
     +gy[R]*Ha[1,2,R[0],R[1],R[2]]+gz[R]*Ha[2,2,R[0],R[1],R[2]])

    gaussian_curv[R] /= L2_norm_grad_narrowband(gx,gy,gz,R)[R]**4

    mean_curv[R] =  (gx[R] * (gx[R]*hessian[0,0,R[0],R[1],R[2]]+gy[R]*hessian[1,0,R[0],R[1],R[2]]+gz[R]*hessian[2,0,R[0],R[1],R[2]]) + \
    gy[R] * (gx[R]*hessian[0,1,R[0],R[1],R[2]]+gy[R]*hessian[1,1,R[0],R[1],R[2]]+gz[R]*hessian[2,1,R[0],R[1],R[2]])\
    + gz[R] * (gx[R]*hessian[0,2,R[0],R[1],R[2]]+gy[R]*hessian[1,2,R[0],R[1],R[2]]+gz[R]*hessian[2,2,R[0],R[1],R[2]])) \
    - (L2_norm_grad_narrowband(gx,gy,gz,R)[R]**2 *  hessian_trace_narrowband(hessian,R))

    mean_curv[R] /= -2*L2_norm_grad_narrowband(gx,gy,gz,R)[R]**3

    return gaussian_curv, mean_curv

def principal_curvatures(K_M, K_G):

    tmp = np.sqrt(np.absolute(K_M**2- K_G))
    k1 = K_M  - tmp
    k2 = K_M  + tmp

    return k1, k2


def save_result(verts,curv,save_path):

        res = np.append(verts,curv[...,None],axis=1)
        np.save(save_path, res)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--mask', help='3D shape binary mask, as NIFTI file', type=str, required = True)
    parser.add_argument('-o', '--output', help='output directory', type=str, default = './Gaussian_curvature_results3D')

    args = parser.parse_args()

    # Example of use : python3 curvatures_narrowband.py -in ./3D_data/stanford_bunny_binary.nii.gz -o /home/karim/Bureau/Courbure/narrow_band

    output_path = args.output

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    shape = nib.load(args.mask).get_data()

    start_time = timeit.default_timer()

    shape, dx, dy, dz = g3D.bbox_3D(shape,5)

    phi, R = phi_narrow(shape,5) ## distance calculation limited to narrow band (not recommended to extract smooth surface mesh)
    phi = local_gaussian_filter(phi, sigma=2) ## smoothing of the level set signed distance function on a narrow band

#################### Computation of  Gaussian and mean curvatures ###############################
    phi_grad, hessian = g3D.hessian(phi)  ### To do in narrowband
    Ha = hessian_adjoint_narrowband(hessian,R)
    Gaussian_curvature, mean_curvature = curvatures_narrowband(phi_grad,Ha,hessian,R)
#################################################################################################
#################### Computation of  principal curvatures #######################################
    K1, K2 = principal_curvatures(mean_curvature, Gaussian_curvature)
#################################################################################################
    # extract explicitly the implicit surface mesh using the scikit-image toolbox

    verts, faces, normals, values = measure.marching_cubes_lewiner(phi, 0.0)
    print(verts.shape)

    ### Affect per-vertex curvature values, with a nearest neighbour interpolation of vertices on the grid
    gaussian_curv = g3D.texture_spline_interpolation3D(verts, Gaussian_curvature)
    mean_curv = g3D.texture_spline_interpolation3D(verts, mean_curvature)
    k1 = g3D.texture_spline_interpolation3D(verts, K1)
    k2 = g3D.texture_spline_interpolation3D(verts, K2)

    elapsed = timeit.default_timer() - start_time
    print("The proposed method takes (in seconds):\n")
    print(elapsed)

    verts = g3D.align_origin_back(verts,dx,dy,dz)
    m = trimesh.Trimesh(vertices=verts, faces=faces)
    m.export(os.path.join(output_path, "surface_mesh.obj"))

    #### Save results as numpy array arrays

    save_result(verts,gaussian_curv,os.path.join(output_path,"gaussian_curv.npy"))
    save_result(verts,mean_curv,os.path.join(output_path,"mean_curv.npy"))
    save_result(verts,k1,os.path.join(output_path,"min_curv.npy"))
    save_result(verts,k2,os.path.join(output_path,"max_curv.npy"))
    save_result(verts,2*gaussian_curv,os.path.join(output_path,"Ricci_scalar.npy"))

    ## Display results

    g3D.display_mesh(verts, faces, normals, gaussian_curv, os.path.join(output_path, "Gaussian_curvature_Makki.png"))
    g3D.display_mesh(verts, faces, normals, 2*gaussian_curv, os.path.join(output_path, "Ricci_scalar_Makki.png"))
    g3D.display_mesh(verts, faces, normals, mean_curv, os.path.join(output_path, "mean_curvature_Makki.png"))
    g3D.display_mesh(verts, faces, normals, k1, os.path.join(output_path, "Minimum_curvature_Makki.png"))
    g3D.display_mesh(verts, faces, normals, k2, os.path.join(output_path, "Maximum_curvature_Makki.png"))
