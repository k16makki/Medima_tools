# -*- coding: utf-8 -*-

"""
  ©
  Author(s): Karim Makki
"""

import visvis as vv
import trimesh
import numpy as np
import nibabel as nib
import os
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
import argparse
import  skfmm
from skimage import measure
import timeit

import slam_curvature as scurv




def hessian(phi):

    phi_grad = np.gradient(phi)
    gaussian_filter(phi_grad[0], sigma=2, output=phi_grad[0])
    gaussian_filter(phi_grad[1], sigma=2, output=phi_grad[1])
    gaussian_filter(phi_grad[2], sigma=2, output=phi_grad[2])
    hessian = np.empty((phi.ndim, phi.ndim) + phi.shape, dtype=phi.dtype)
    for k, grad_k in enumerate(phi_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return phi_grad, hessian


#### Compute the adjoint of the Hessian matrix
def hessian_adjoint(hessian):

    Ha = np.zeros(hessian.shape)
    Ha[0,0,...] = hessian[1,1,...]*hessian[2,2,...] - hessian[1,2,...]*hessian[2,1,...]
    Ha[0,1,...] = hessian[1,2,...]*hessian[2,0,...] - hessian[1,0,...]*hessian[2,2,...]
    Ha[0,2,...] = hessian[1,0,...]*hessian[2,1,...] - hessian[1,1,...]*hessian[2,0,...]

    Ha[1,0,...] = hessian[0,2,...]*hessian[2,1,...] - hessian[0,1,...]*hessian[2,2,...]
    Ha[1,1,...] = hessian[0,0,...]*hessian[2,2,...] - hessian[0,2,...]*hessian[2,0,...]
    Ha[1,2,...] = hessian[0,1,...]*hessian[2,0,...] - hessian[0,0,...]*hessian[2,1,...]

    Ha[2,0,...] = hessian[0,1,...]*hessian[1,2,...] - hessian[0,2,...]*hessian[1,1,...]
    Ha[2,1,...] = hessian[1,0,...]*hessian[0,2,...] - hessian[0,0,...]*hessian[1,2,...]
    Ha[2,2,...] = hessian[0,0,...]*hessian[1,1,...] - hessian[0,1,...]*hessian[1,0,...]

    return Ha

def norm_grad(gx,gy,gz):

    norm_grad =  np.sqrt(np.power(gx,2)+np.power(gy,2)+np.power(gz,2))
    norm_grad[np.where(norm_grad==0)]=1

    return  norm_grad

def Gaussian_curvature(phi_grad,Ha):

    gx, gy, gz = phi_grad

    gaussian_curv = gx * (gx*Ha[0,0,...]+gy*Ha[1,0,...]+gz*Ha[2,0,...]) + gy * (gx*Ha[0,1,...]+gy*Ha[1,1,...]+gz*Ha[2,1,...])\
    + gz * (gx*Ha[0,2,...]+gy*Ha[1,2,...]+gz*Ha[2,2,...])

    norm = norm_grad(gx,gy,gz)
    np.divide(gaussian_curv,np.power(norm,4),gaussian_curv)

    return gaussian_curv

def bbox_3D(mask,depth):

    x = np.any(mask, axis=(1, 2))
    y = np.any(mask, axis=(0, 2))
    z = np.any(mask, axis=(0, 1))
    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return mask[xmin-depth:xmax+depth,ymin-depth:ymax+depth,zmin-depth:zmax+depth]


## signed geodesic distance
def phi(mask):

    phi_ext = skfmm.distance(np.max(mask)-mask)
    phi_int = skfmm.distance(mask)

    return  phi_ext - phi_int

## signed Euclidean distance
def phi_Euclidean(mask):

    phi_ext = ndimage.distance_transform_edt(np.max(mask)-mask)
    phi_int = ndimage.distance_transform_edt(mask)

    return phi_ext - phi_int



def display_mesh(verts, faces, normals, texture, save_path):

    mesh = vv.mesh(verts, faces, normals, texture, verticesPerFace=3)
    f = vv.gca()
    mesh.colormap = vv.CM_JET
    vv.callLater(1.0, vv.screenshot, save_path, vv.gcf(), sf=2)
    vv.colorbar()
    f.axis.visible = False
    vv.use().Run()

    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--mask', help='3D shape binary mask, as NIFTI file', type=str, required = True)
    parser.add_argument('-o', '--output', help='output directory', type=str, default = './Gaussian_curvature_results')

    args = parser.parse_args()

    # Example of use : python3 fast_Gaussian_curvature_3D.py -in ./data/stanford_bunny_binary.nii.gz

    output_path = args.output

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    shape = nib.load(args.mask).get_data()

    start_time = timeit.default_timer()

    shape = bbox_3D(shape,5)

    phi = phi(shape) ## geodesic signed distance

    gaussian_filter(phi, sigma=2, output=phi) ## smoothing of the level set signed distance function

#################### Computation of  Gaussian curvature ###################
    phi_grad, hessian = hessian(phi)
    Ha = hessian_adjoint(hessian)
    Gaussian_curvature = Gaussian_curvature(phi_grad, Ha)

############################################################################

    elapsed = timeit.default_timer() - start_time
    print("The proposed method takes:\n")
    print(elapsed)

    verts, faces, normals, values = measure.marching_cubes_lewiner(phi, 0.0) # surface mesh
    print(verts.shape)
    m = trimesh.Trimesh(vertices=verts, faces=faces)
    #m.export(output_path+'/surface_mesh.ply')
    m.export(output_path+'/surface_mesh.obj')

    texture = Gaussian_curvature[verts[:,0].astype(int),verts[:,1].astype(int),verts[:,2].astype(int)]
    display_mesh(verts, faces, normals, texture, output_path + '/Gaussian_curature.png')

# To compare results with the mean curvature based on the estimation of principal curvature and derivatives, please uncomment the following block

'''
    # Comptue estimations of principal curvatures

    m = trimesh.load_mesh(output_path+'/surface_mesh.obj')

    start_time = timeit.default_timer()
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(m)

###############################################################################
# Comptue Gaussian curvature from principal curvatures
    gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]

    elapsed = timeit.default_timer() - start_time

    print("The Rusinkiewicz method takes:\n")
    print(elapsed)

    display_mesh(m.vertices, m.faces, m.vertex_normals, gaussian_curv, output_path + '/Gaussian_curature_Rusinkiewicz.png')
'''
