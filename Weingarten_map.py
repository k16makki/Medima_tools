# -*- coding: utf-8 -*-

"""
  ©
  Author: Karim Makki
"""

import visvis as vv
import trimesh
import numpy as np
import os
import argparse
from scipy.ndimage.filters import gaussian_filter
import  skfmm
import nibabel as nib
from skimage import measure

import fast_Gaussian_curvature_3D as g3D


#### Compute the Hessian determinant using the rule of Sarrus

def hessian_determinant(h):

    tmp1 = h[0,0,...]* h[1,1,...]*h[2,2,...] + h[0,1,...]* h[1,2,...]*h[2,0,...] +h[0,2,...]* h[1,0,...]*h[2,1,...]
    tmp2 = h[0,1,...]* h[1,0,...]*h[2,2,...] + h[0,0,...]* h[1,2,...]*h[2,1,...] +h[0,2,...]* h[1,1,...]*h[2,0,...]

    return tmp1 - tmp2



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--mask', help='3D shape binary mask, as NIFTI file', type=str, required = True)
    parser.add_argument('-o', '--output', help='output directory', type=str, default = './principal_curvature_eigen_results')


    args = parser.parse_args()


    ## Example of use: time python3 Weingarten_map.py -in /home/karim/Bureau/Courbure/data/stanford_bunny_binary.nii.gz -o /home/karim/Bureau/Courbure/test/eigen_results


    output_path = args.output

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    shape = nib.load(args.mask).get_data()

    shape, dx, dy, dz = g3D.bbox_3D(shape,7)
    phi = g3D.phi(shape) ## signed geodesic distance

    gaussian_filter(phi, sigma=2, output=phi) ## the smoothing kernel should be the same everywhere

    verts, faces, normals, values = measure.marching_cubes_lewiner(phi, 0.0)#, spacing=(dx,dy,dz), gradient_direction='descent')


    Hessian = g3D.hessian(phi)[1]


    Hessian_determinant = hessian_determinant(Hessian)
    det = g3D.texture_spline_interpolation3D(verts, Hessian_determinant)

    ### Compute sorted eigenvalues of the Hessian matrix ###############

    Hessian = np.einsum('lmijk->ijklm', Hessian)
    eigenValues, eigenVectors = np.linalg.eig(Hessian)
    eigenValues = np.sort(eigenValues)

    ####################################################################

    #lamda1 = g3D.texture_nearest_neigh_interpolation3D(verts, eig_vals_sorted[...,0])
    lamda1 = g3D.texture_spline_interpolation3D(verts, eigenValues[...,0])
    lamda2 = g3D.texture_spline_interpolation3D(verts, eigenValues[...,1])
    lamda3 = g3D.texture_spline_interpolation3D(verts, eigenValues[...,2])

    verts = g3D.align_origin(verts,dx,dy,dz) ### re-align origin

    m = trimesh.Trimesh(vertices=verts, faces=faces)

    m.export(os.path.join(output_path, "surface_mesh.obj"))
    #

    g3D.display_mesh(verts, faces, normals, lamda1, os.path.join(output_path, "lambda1.png"))
    g3D.display_mesh(verts, faces, normals, lamda2, os.path.join(output_path, "lambda2.png"))
    g3D.display_mesh(verts, faces, normals, lamda3, os.path.join(output_path, "lambda3.png"))

    g3D.display_mesh(verts, faces, normals, (lamda1+lamda2+lamda3)/2, os.path.join(output_path, "mean_curvature1.png"))
    g3D.display_mesh(verts, faces, normals, lamda1+lamda2+lamda3, os.path.join(output_path, "Laplacian.png"))
    g3D.display_mesh(verts, faces, normals, det, os.path.join(output_path, "Hessian_determinant_Sarrus.png"))
    #g3D.display_mesh(verts, faces, normals, lamda1*lamda3, os.path.join(output_path, "gaussian_curvature1.png"))
