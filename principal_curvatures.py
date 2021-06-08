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


## Import tools for computing curvature on explicit surfaces (for comparison purposes)
import slam_curvature as scurv
import fast_Gaussian_curvature_3D as g3D


def principal_curvatures(K_M,K_G):

    tmp = np.sqrt(np.absolute(K_M[:,3]**2- K_G[:,3]))

    k1 = K_M[:,3] - tmp
    k2 = K_M[:,3] + tmp



    return k1, k2



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mean', '--mean_curv', help='mean curvature results, (N x 4) npy file containing coordinates of N vertices and curvature\
     values (last column) ', type=str, required = True)
    parser.add_argument('-gauss', '--gaussian_curv', help='gaussian curvature results, (N x 4) npy file containing coordinates of N vertices and curvature\
    values (last column) ', type=str, required = True)
    parser.add_argument('-in', '--mask', help='3D shape binary mask, as NIFTI file', type=str, required = True)
    parser.add_argument('-o', '--output', help='output directory', type=str, default = './principal_curvature_results')


    args = parser.parse_args()

    ##Warning: the signed distance function shoud be the same as that (unique) used for estiationg both mean and gaussian curvature


    ## Example of use: time python3 principal_curvatures.py -mean /home/karim/Bureau/Courbure/test/mean_curv.npy -gauss
    #/home/karim/Bureau/Courbure/test/gaussian_curv.npy -in /home/karim/Bureau/Courbure/data/sphere_r30_bosse.nii.gz -o /home/karim/Bureau/Courbure/test/results


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


    K_M = np.load(args.mean_curv)
    K_G = np.load(args.gaussian_curv)


    k1,k2 = principal_curvatures(K_M,K_G)

    #gaussian_filter(k1, sigma=2, output=k1)
    #gaussian_filter(k2, sigma=2, output=k2)

    g3D.display_mesh(verts, faces, normals, k1, os.path.join(output_path, "Principal_curvature1.png"))
    g3D.display_mesh(verts, faces, normals, k2, os.path.join(output_path, "Principal_curvature2.png"))


# #######################################################################################################################################
# ##### To compare results with the Rusinkiewicz (v1) principal curvatures, please uncomment the following block ########################

    m = trimesh.load_mesh(os.path.join(output_path, "surface_mesh.obj"))
    # Comptue estimations of principal curvatures
    PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(m)

    g3D.display_mesh(m.vertices, m.faces, m.vertex_normals, PrincipalCurvatures[0, :], os.path.join(output_path, "Principal_curvature1_Rusinkiewicz.png"))
    g3D.display_mesh(m.vertices, m.faces, m.vertex_normals, PrincipalCurvatures[1, :], os.path.join(output_path, "Principal_curvature2_Rusinkiewicz.png"))
#########################################################################################################################################
