# -*- coding: utf-8 -*-

"""
  ©
  Author: Karim Makki
"""

import numpy as np
import os
import argparse
import slam_curvature as scurv
import fast_Gaussian_curvature_3D as g3D
import Gauss_curv_4_Freesurfer_output as GFS


def principal_curvatures(K_M,K_G):

    tmp = np.sqrt(np.absolute(K_M[:,3]**2- K_G[:,3]))
    k1 = K_M[:,3]  - tmp
    k2 = K_M[:,3]  + tmp

    return k1, k2



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-mean', '--mean_curv', help='mean curvature results, (N x 4) npy file containing coordinates of N vertices and curvature\
     values (last column) ', type=str, required = True)
    parser.add_argument('-gauss', '--gaussian_curv', help='gaussian curvature results, (N x 4) npy file containing coordinates of N vertices and curvature\
    values (last column) ', type=str, required = True)
    parser.add_argument('-m', '--mesh', help='mesh, as GIFTI file', type=str, required = True)
    parser.add_argument('-o', '--output', help='output directory', type=str, default = './principal_curvature_results')


    args = parser.parse_args()

    ## Example of use: python3 principal_curvatures_4_Freesurfer_output.py -mean /home/karim/Bureau/Courbure/mean_curv.npy
    #-gauss /home/karim/Bureau/Courbure/gaussian_curv.npy -m /home/karim/Bureau/Courbure/data/Guillaume_data/rh.white.gii
    # -o /home/karim/Bureau/Courbure/principal_curvatures


    output_path = args.output

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #mesh = trimesh.load_mesh(args.mesh)

    mesh = GFS.load_mesh(args.mesh)


    K_M = np.load(args.mean_curv)
    K_G = np.load(args.gaussian_curv)


    k2,k1 = principal_curvatures(K_M,K_G)

    g3D.display_mesh(mesh.vertices, mesh.faces, mesh.vertex_normals, k1, os.path.join(output_path, "Principal_curvature1.png"))
    g3D.display_mesh(mesh.vertices, mesh.faces, mesh.vertex_normals, k2, os.path.join(output_path, "Principal_curvature2.png"))
