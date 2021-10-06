# -*- coding: utf-8 -*-

"""
  ©
  Author: Karim Makki
"""

#import visvis as vv
import trimesh
import numpy as np
import nibabel as nib
import os
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
import argparse
from skimage import measure
import timeit
import fast_Gaussian_curvature_3D as g3D


## Import tools for computing curvature on explicit surfaces (for comparison purposes)
import slam_curvature as scurv
import CurvatureCubic as ccurv
import CurvatureWpF as WpFcurv
import CurvatureISF as ISFcurv
from trimesh import curvature
import DiffGeoOps as diffgeo



def hessian_trace(hessian):

    return hessian[0,0,...] + hessian[1,1,...] + hessian[2,2,...]



def mean_curvature(phi_grad,hessian):

    gx, gy, gz = phi_grad

    norm = g3D.L2_norm_grad(gx,gy,gz)
    gx /= norm
    gy /= norm
    gz /= norm

    mean_curv =  (gx * (gx*hessian[0,0,...]+gy*hessian[1,0,...]+gz*hessian[2,0,...]) + gy * (gx*hessian[0,1,...]+gy*hessian[1,1,...]+gz*hessian[2,1,...])\
    + gz * (gx*hessian[0,2,...]+gy*hessian[1,2,...]+gz*hessian[2,2,...])) - (g3D.L2_norm_grad(gx,gy,gz)**2 *  hessian_trace(hessian))

    np.divide(mean_curv,-2*g3D.L2_norm_grad(gx,gy,gz)**3,mean_curv)

    return mean_curv



def divergence_formula(phi):

    g_x,g_y,g_z = np.gradient(phi)
    #smoothing of gradient vector field
    #gaussian_filter(g_x, sigma=2, output=g_x)
    #gaussian_filter(g_y, sigma=2, output=g_y)
    #gaussian_filter(g_z, sigma=2, output=g_z)
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



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--mask', help='3D shape binary mask, as NIFTI file', type=str, required = True)
    parser.add_argument('-o', '--output', help='output directory', type=str, default = './mean_curvature_results3D')
    parser.add_argument('-dmap', '--dmap', help='distance_map: 0 if Euclidean, 1 if geodesic distance map, and 2 if binary step function', type=int, default = 1)

    args = parser.parse_args()

    # Example of use : python3 fast_mean_curvature_3D.py -in ./3D_data/stanford_bunny_binary.nii.gz

    output_path = args.output

    if not os.path.exists(output_path):
        os.makedirs(output_path)


    shape = nib.load(args.mask).get_data()

    start_time = timeit.default_timer()

    shape, dx, dy, dz = g3D.bbox_3D(shape)

    if (args.dmap == 1):

        phi = g3D.phi(shape) ## signed geodesic distance

    elif (args.dmap == 2):

        phi = g3D.phi_binary(shape) ## binary step function

    else:

        phi = g3D.phi_Euclidean(shape) ## signed Euclidean distance

    gaussian_filter(phi, sigma=2, output=phi) ## smoothing of the level set signed distance function

    ########## Compute mean curvature ###################

    #curvature = divergence_formula(phi)   ### uncomment this line if you would like to run the divergence formula and comment the two following lines

    phi_grad, hessian = g3D.hessian(phi)
    mean_curvature = mean_curvature(phi_grad,hessian)

    ######################################################

    elapsed = timeit.default_timer() - start_time

    print("The proposed method takes (in seconds):\n")
    print(elapsed)

    # extract explicitly the implicit surface mesh using the scikit-image toolbox

    verts, faces, normals, values = measure.marching_cubes_lewiner(phi, 0.0)#, gradient_direction='descent')

    print(verts.shape)

    # Affect per-vertex curvature values, by interpolation
    #mean_curv = g3D.texture_mean_avg_interpolation3D(verts, mean_curvature)
    mean_curv = g3D.texture_spline_interpolation3D(verts, mean_curvature)

    verts = g3D.align_origin_back(verts,dx,dy,dz)

    m = trimesh.Trimesh(vertices=verts, faces=faces)
    #m.export(output_path+'/surface_mesh.ply')
    m.export(os.path.join(output_path, "surface_mesh.obj"))

    #print(np.min(mean_curv),np.max(mean_curv), np.mean(mean_curv))

    #### Save results as numpy array

    res = np.append(verts,mean_curv[...,None],axis=1)
    np.save(os.path.join(output_path, "mean_curv.npy"), res)
    print(res.shape)

    ## Display result

    #g3D.display_mesh(verts, faces, normals, None, os.path.join(output_path, "surface_makki.png"))
    g3D.display_mesh(verts, faces, normals, mean_curv, os.path.join(output_path, "mean_curvature_Makki.png"))


##To compare results with other methods defining the surface explicitly, please uncomment one of the following blocks #################


# #######################################################################################################################################
# ################### To compare results with the Trimesh mean curvature, please uncomment this block ###################################
#
#     m = trimesh.load_mesh(os.path.join(output_path, "surface_mesh.obj"))
#
#     start_time = timeit.default_timer()
#
#     #tr_mean_curv = curvature.discrete_mean_curvature_measure(m, m.vertices, 2)
#     tr_mean_curv = curvature.discrete_mean_curvature_measure(m, m.vertices, 1.0)
#
#     elapsed = timeit.default_timer() - start_time
#
#     print("The Trimesh method takes (in seconds):\n")
#
#     print(elapsed)
#
#     g3D.display_mesh(m.vertices, m.faces, m.vertex_normals, tr_mean_curv, os.path.join(output_path, "Mean_curvature_Trimesh.png"))
#
# #########################################################################################################################################
#
# #######################################################################################################################################
# ################## To compare results with the Rusinkiewicz (v1) mean curvature, please uncomment this block ##########################
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
#     #print(np.min(mean_curv),np.max(mean_curv), np.sqrt(np.absolute(np.mean(mean_curv)-(1/R))))
#     #gaussian_filter(mean_curv, sigma=1, output=mean_curv)
#     g3D.display_mesh(m.vertices, m.faces, m.vertex_normals, mean_curv, os.path.join(output_path, "mean_curvature_Rusinkiewicz_v1.png"))
# #########################################################################################################################################
#
#
# #########################################################################################################################################
# ##### To compare results with the Rusinkiewicz (v2) mean curvature, please uncomment this block #########################################
# ########################### Note that the second version is quite  faster than the first ################################################
#
#     m = trimesh.load_mesh(os.path.join(output_path, "surface_mesh.obj"))
#
#     start_time = timeit.default_timer()
#
#     #K,H,VN = WpFcurv.GetCurvatures(m.vertices,m.faces)
#     mean_curv = WpFcurv.GetCurvatures(m.vertices,m.faces)[1]
#
#     elapsed = timeit.default_timer() - start_time
#
#     print("The Rusinkiewicz method v2 takes (in seconds):\n")
#     print(elapsed)
#
#     #gaussian_filter(mean_curv, sigma=1, output=gaussian_curv)
#     g3D.display_mesh(m.vertices, m.faces, m.vertex_normals, mean_curv, os.path.join(output_path, "mean_curvature_Rusinkiewicz_v2.png"))
# ##########################################################################################################################################
#
#
# #########################################################################################################################################
# ############## To compare results with those of the cubic order algorithm, please uncomment this block ##################################
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
#     g3D.display_mesh(m.vertices, m.faces, m.vertex_normals, mean_curv, os.path.join(output_path, "mean_curvature_cubic_order.png"))
# ##########################################################################################################################################
#
# #########################################################################################################################################
# ############## To compare results with the iterative fitting method, please uncomment this block ########################################
#
#     m = trimesh.load_mesh(os.path.join(output_path, "surface_mesh.obj"))
#
#     start_time = timeit.default_timer()
#
#     mean_curv = ISFcurv.CurvatureISF2(m.vertices,m.faces)[1]
#
#     elapsed = timeit.default_timer() - start_time
#
#     print("The iterative fitting method takes (in seconds):\n")
#     print(elapsed)
#
#     g3D.display_mesh(m.vertices, m.faces, m.vertex_normals, mean_curv, os.path.join(output_path, "mean_curvature_iterative_fitting.png"))
# ##########################################################################################################################################
#
# #########################################################################################################################################
# ############## To compare results with the method of Meyer, please uncomment this block #################################################
#
#     m = trimesh.load_mesh(os.path.join(output_path, "surface_mesh.obj"))
#
#     start_time = timeit.default_timer()
#
#     A_mixed, mean_curvature_normal_operator_vector = diffgeo.calc_A_mixed(m.vertices, m.faces)
#     mean_curv = diffgeo.get_mean_curvature(mean_curvature_normal_operator_vector)
#
#     elapsed = timeit.default_timer() - start_time
#
#     print("The method of Meyer takes (in seconds):\n")
#     print(elapsed)
#
#     g3D.display_mesh(m.vertices, m.faces, m.vertex_normals, mean_curv, os.path.join(output_path, "mean_curvature_Meyer.png"))
# ##########################################################################################################################################
