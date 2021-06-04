# -*- coding: utf-8 -*-

"""
  ©
  Author: Karim Makki
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

## Import tools for computing curvature on explicit surfaces (for comparison purposes)
import slam_curvature as scurv
import CurvatureCubic as ccurv
import CurvatureWpF as WpFcurv



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

## Binary step function, equivalent to the signed distance funcion, as it satisfies |\nabla \phi | = 1, at least in a vincinity
## of the zero level set
def phi_binary(mask):

    phi_bin = np.ones(mask.shape)
    phi_bin[np.where(mask != 0)] = -1

    return  phi_bin


def curvature(phi):

    g_x,g_y,g_z = np.gradient(phi)
    #smoothing of gradient vector field
    gaussian_filter(g_x, sigma=2, output=g_x)
    gaussian_filter(g_y, sigma=2, output=g_y)
    gaussian_filter(g_z, sigma=2, output=g_z)
    norm_grad =  np.sqrt(np.power(g_x,2)+np.power(g_y,2)+np.power(g_z,2))
    norm_grad[np.where(norm_grad==0)]=1
    np.divide(g_x,norm_grad,g_x)
    np.divide(g_y,norm_grad,g_y)
    np.divide(g_z,norm_grad,g_z)
    g_xx, g_yx , g_zx = np.gradient(g_x)
    g_xy, g_yy , g_zy = np.gradient(g_y)
    g_xz, g_yz , g_zz = np.gradient(g_z)
    gaussian_filter(g_xx, sigma=2, output=g_xx)
    gaussian_filter(g_yy, sigma=2, output=g_yy)
    gaussian_filter(g_zz, sigma=2, output=g_zz)

    return  0.5*(g_xx + g_yy + g_zz)

def display_mesh(verts, faces, normals, texture, save_path):

    mesh = vv.mesh(verts, faces, normals, texture, verticesPerFace=3)
    f = vv.gca()
    mesh.colormap = vv.CM_JET
    vv.callLater(1.0, vv.screenshot, save_path, vv.gcf(), sf=2)
    vv.colorbar()
    f.axis.visible = False
    vv.use().Run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--mask', help='3D shape binary mask, as NIFTI file', type=str, required = True)
    parser.add_argument('-o', '--output', help='output directory', type=str, default = './curvature_results')
    parser.add_argument('-dmap', '--dmap', help='distance_map: 0 if Euclidean, 1 if geodesic distance map, and 2 if binary step function', type=int, default = 0)

    args = parser.parse_args()

    # Example of use : python3 fast_mean_curvature_3D.py -in ./3D_data/stanford_bunny_binary.nii.gz

    output_path = args.output

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    shape = nib.load(args.mask).get_data()

    start_time = timeit.default_timer()

    shape = bbox_3D(shape,5)

    if (args.dmap == 1):

        phi = phi(shape) ## signed geodesic distance

    elif (args.dmap == 2):

        phi = phi_binary(shape) ## binary step function

    else:

        phi = phi_Euclidean(shape) ## signed Euclidean distance

    gaussian_filter(phi, sigma=2, output=phi) ## smoothing of the level set signed distance function

    curvature = curvature(phi)

    elapsed = timeit.default_timer() - start_time

    print("The proposed method takes:\n")
    print(elapsed)

    # extract explicitly the implicit surface mesh using the scikit-image toolbox

    verts, faces, normals, values = measure.marching_cubes_lewiner(phi, 0.0, gradient_direction='descent')

    print(verts.shape)
    m = trimesh.Trimesh(vertices=verts, faces=faces)
    #m.export(output_path+'/surface_mesh.ply')
    m.export(os.path.join(output_path, "surface_mesh.obj"))


    texture = curvature[verts[:,0].astype(int),verts[:,1].astype(int),verts[:,2].astype(int)]
    #print(np.min(texture),np.max(texture))
    display_mesh(verts, faces, normals, texture, os.path.join(output_path, "mean_curature.png"))

#####To compare results with other methods defining the surface explicitly, please comment/uncomment the following blocks ###############

# #######################################################################################################################################
# ##### To compare results with the Rusinkiewicz (v1) mean curvature, please uncomment the following block ##########################
#
#     m = trimesh.load_mesh(os.path.join(output_path, "surface_mesh.obj"))
#
#     start_time = timeit.default_timer()
#     # Comptue estimations of principal curvatures
#     PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(m)
#     mean_curv = 0.5*(PrincipalCurvatures[0, :] + PrincipalCurvatures[1, :])
#
#     elapsed = timeit.default_timer() - start_time
#
#     print("The Rusinkiewicz method v1 takes (in seconds):\n")
#     print(elapsed)
#
#     #gaussian_filter(mean_curv, sigma=1, output=mean_curv)
#     display_mesh(m.vertices, m.faces, m.vertex_normals, mean_curv, os.path.join(output_path, "mean_curvature_Rusinkiewicz_v1.png"))
# #########################################################################################################################################


#########################################################################################################################################
##### To compare results with the Rusinkiewicz (v2) mean curvature, please uncomment the following block ############################
########################### Note that the second version is quite  faster than the first ################################################

    m = trimesh.load_mesh(os.path.join(output_path, "surface_mesh.obj"))

    start_time = timeit.default_timer()

    #K,H,VN = WpFcurv.GetCurvatures(m.vertices,m.faces)
    mean_curv = WpFcurv.GetCurvatures(m.vertices,m.faces)[1]

    elapsed = timeit.default_timer() - start_time

    print("The Rusinkiewicz method v2 takes (in seconds):\n")
    print(elapsed)

    #gaussian_filter(gaussian_curv, sigma=1, output=gaussian_curv)
    display_mesh(m.vertices, m.faces, m.vertex_normals, mean_curv, os.path.join(output_path, "mean_curvature_Rusinkiewicz_v2.png"))
##########################################################################################################################################


# #########################################################################################################################################
# ##### To compare results with those of the cubic order algorithm, please uncomment the following block ##################################
#
#     m = trimesh.load_mesh(os.path.join(output_path, "surface_mesh.obj"))
#
#     start_time = timeit.default_timer()
#
#     #K,H,VN = ccurv.CurvatureCubic(m.vertices,m.faces)
#     mean_curv = ccurv.CurvatureCubic(m.vertices,m.faces)[1]
#
#     elapsed = timeit.default_timer() - start_time
#
#     print("The cubic order algorithm takes (in seconds):\n")
#     print(elapsed)
#
#     #gaussian_filter(mean_curv, sigma=1, output=mean_curv)
#     display_mesh(m.vertices, m.faces, m.vertex_normals, mean_curv, os.path.join(output_path, "mean_curvature_cubic_order.png"))
# ##########################################################################################################################################
