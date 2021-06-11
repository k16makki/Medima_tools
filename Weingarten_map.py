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


    ## Example of use: time python3 Weingarten_map.py -in /home/karim/Bureau/Courbure/data/cortex.nii.gz -o /home/karim/Bureau/Courbure/test/eigen_results


    output_path = args.output

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    shape = nib.load(args.mask).get_data()

    shape = g3D.bbox_3D(shape,5)
    phi = g3D.phi(shape) ## signed geodesic distance

    gaussian_filter(phi, sigma=2, output=phi) ## the smoothing kernel should be the same everywhere

    verts, faces, normals, values = measure.marching_cubes_lewiner(phi, 0.0)#, spacing=(dx,dy,dz), gradient_direction='descent')

    m = trimesh.Trimesh(vertices=verts, faces=faces)

    m.export(os.path.join(output_path, "surface_mesh.obj"))


    Hessian = g3D.hessian(phi)[1]

    #Hessian = g3D.hessian_adjoint(Hessian)

    Hessian_determinant = hessian_determinant(Hessian)

    ### Compute sorted eigenvalues of the Hessian matrix ###############

    Hessian = np.einsum('lmijk->ijklm', Hessian)
    eigenValues, eigenVectors = np.linalg.eig(Hessian)
    eig_vals_sorted = np.sort(eigenValues)

    ####################################################################

    lamda1 = g3D.texture_interpolation3D(verts, eig_vals_sorted[...,0])
    lamda2 = g3D.texture_interpolation3D(verts, eig_vals_sorted[...,1])
    lamda3 = g3D.texture_interpolation3D(verts, eig_vals_sorted[...,2])


    #det = Hessian_determinant[np.rint(verts[:,0]).astype(int),np.rint(verts[:,1]).astype(int),np.rint(verts[:,2]).astype(int)]


    g3D.display_mesh(verts, faces, normals, lamda1, os.path.join(output_path, "lambda1.png"))
    g3D.display_mesh(verts, faces, normals, lamda2, os.path.join(output_path, "lambda2.png"))
    g3D.display_mesh(verts, faces, normals, lamda3, os.path.join(output_path, "lambda3.png"))

    g3D.display_mesh(verts, faces, normals, (lamda2+lamda3)/2, os.path.join(output_path, "mean_curvature1.png"))
    #g3D.display_mesh(verts, faces, normals, det, os.path.join(output_path, "Hessian_determinant_Sarrus.png"))
    g3D.display_mesh(verts, faces, normals, lamda2*lamda3, os.path.join(output_path, "gaussian_curvature1.png"))

    #error = np.sqrt(np.absolute(det**2 - (lamda1*lamda2*lamda3)**2 ))

    #print(np.min(error),np.max(error),np.mean(error))
