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
import CurvatureISF as ISFcurv
from trimesh import curvature
import DiffGeoOps as diffgeo


def load_mesh(gii_file):
    """
    load gifti_file and create a trimesh object
    :param gifti_file: str, path to the gifti file
    :return: the corresponding trimesh object
    """
    g = nib.gifti.read(gii_file)
    vertices, faces = g.getArraysFromIntent(
        nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET'])[0].data, \
        g.getArraysFromIntent(
            nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])[0].data
    metadata = g.get_meta().metadata
    metadata['filename'] = gii_file

    return trimesh.Trimesh(faces=faces, vertices=vertices,
                           metadata=metadata, process=False)


def map_coordinates(verts,aff):

    coords = np.zeros(verts.shape)
    coords[:,0] = aff[0,0]*verts[:,0] + aff[0,1]*verts[:,1] + aff[0,2]*verts[:,2] + aff[0,3]
    coords[:,1] = aff[1,0]*verts[:,0] + aff[1,1]*verts[:,1] + aff[1,2]*verts[:,2] + aff[1,3]
    coords[:,2] = aff[2,0]*verts[:,0] + aff[2,1]*verts[:,1] + aff[2,2]*verts[:,2] + aff[2,3]

    return coords


def hessian(phi):

    phi_grad = np.gradient(phi)
    hessian = np.empty((phi.ndim, phi.ndim) + phi.shape, dtype=phi.dtype)
    for k, grad_k in enumerate(phi_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl

    return phi_grad, hessian


#### Compute the adjoint of the Hessian matrix (faster than the numpy version defined below)
#### Reference: Ron Goldman, Curvature formulas for implicit curves and surfaces, Computer Aided Geometric Design 22 (2005) 632–658

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

#### Compute the adjoint of the Hessian matrix
def hessian_adjoint_numpy(hessian):

    Ha = np.zeros(hessian.shape)
    Ha[0,0,...] = np.subtract(np.multiply(hessian[1,1,...],hessian[2,2,...]) , np.multiply(hessian[1,2,...],hessian[2,1,...]))
    Ha[0,1,...] = np.subtract(np.multiply(hessian[1,2,...],hessian[2,0,...]) , np.multiply(hessian[1,0,...],hessian[2,2,...]))
    Ha[0,2,...] = np.subtract(np.multiply(hessian[1,0,...],hessian[2,1,...]) , np.multiply(hessian[1,1,...],hessian[2,0,...]))

    Ha[1,0,...] = np.subtract(np.multiply(hessian[0,2,...],hessian[2,1,...]) , np.multiply(hessian[0,1,...],hessian[2,2,...]))
    Ha[1,1,...] = np.subtract(np.multiply(hessian[0,0,...],hessian[2,2,...]) , np.multiply(hessian[0,2,...],hessian[2,0,...]))
    Ha[1,2,...] = np.subtract(np.multiply(hessian[0,1,...],hessian[2,0,...]) , np.multiply(hessian[0,0,...],hessian[2,1,...]))

    Ha[2,0,...] = np.subtract(np.multiply(hessian[0,1,...],hessian[1,2,...]) , np.multiply(hessian[0,2,...],hessian[1,1,...]))
    Ha[2,1,...] = np.subtract(np.multiply(hessian[1,0,...],hessian[0,2,...]) , np.multiply(hessian[0,0,...],hessian[1,2,...]))
    Ha[2,2,...] = np.subtract(np.multiply(hessian[0,0,...],hessian[1,1,...]) , np.multiply(hessian[0,1,...],hessian[1,0,...]))

    return Ha

def L2_norm_grad(gx,gy,gz):

    norm_grad =  np.sqrt(np.square(gx)+np.square(gy)+np.square(gz))
    gaussian_filter(norm_grad, sigma=1, output=norm_grad)
    norm_grad[np.where(norm_grad==0)]=1 # just to avoid dividing by zero

    return  norm_grad


def norm_grad_Taxicab(gx,gy,gz):

    norm_grad =  np.absolute(gx)+np.absolute(gy)+np.absolute(gz)
    gaussian_filter(norm_grad, sigma=1, output=norm_grad)
    norm_grad[np.where(norm_grad==0)]= 1

    return  norm_grad


def L_infinity_norm_grad(gx,gy,gz):

    norm_grad =  np.maximum(np.maximum(np.absolute(gx),np.absolute(gy)), np.absolute(gz))
    gaussian_filter(norm_grad, sigma=1, output=norm_grad)
    norm_grad[np.where(norm_grad==0)]= 1

    return  norm_grad


def Gaussian_curvature(phi_grad,Ha):

    gx, gy, gz = phi_grad

    gaussian_curv =  gx * (gx*Ha[0,0,...]+gy*Ha[1,0,...]+gz*Ha[2,0,...]) + gy * (gx*Ha[0,1,...]+gy*Ha[1,1,...]+gz*Ha[2,1,...])\
    + gz * (gx*Ha[0,2,...]+gy*Ha[1,2,...]+gz*Ha[2,2,...])
    #np.divide(gaussian_curv,np.power(L2_norm_grad(gx,gy,gz),4),gaussian_curv)
    np.divide(gaussian_curv,L2_norm_grad(gx,gy,gz)**4,gaussian_curv)
    #gaussian_filter(gaussian_curv, sigma=1, output=gaussian_curv)

    return gaussian_curv


def Hessian_adjoint_curvature(phi_grad,Ha):

    gx, gy, gz = phi_grad

    curvature =  gx * (gx*Ha[0,0,...]+gy*Ha[1,0,...]+gz*Ha[2,0,...]) + gy * (gx*Ha[0,1,...]+gy*Ha[1,1,...]+gz*Ha[2,1,...])\
    + gz * (gx*Ha[0,2,...]+gy*Ha[1,2,...]+gz*Ha[2,2,...])
    np.divide(curvature,L2_norm_grad(gx,gy,gz)**3,curvature)
    #gaussian_filter(curvature, sigma=1, output=curvature)

    return curvature


def bbox_3D(mask,depth):

    x = np.any(mask, axis=(1, 2))
    y = np.any(mask, axis=(0, 2))
    z = np.any(mask, axis=(0, 1))
    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return mask[xmin-depth:xmax+depth,ymin-depth:ymax+depth,zmin-depth:zmax+depth]


## signed geodesic distance function for the implicit surface
def phi_v1(mask):

    phi_ext = skfmm.distance(np.max(mask)-mask)
    phi_int = skfmm.distance(mask)

    return  phi_ext - phi_int


def phi(mask):

    tmp = np.ones(mask.shape)
    tmp[np.where(mask!=0)]= -1

    return  skfmm.distance(tmp)


## signed Euclidean distance
def phi_Euclidean(mask):

    mask[np.where(mask!=0)] = 1
    phi_ext = ndimage.distance_transform_edt(1-mask)
    phi_int = ndimage.distance_transform_edt(mask)

    return phi_ext - phi_int

## Binary step function, equivalent to the signed distance funcion, as it satisfies |\nabla \phi | = 1, but exclusively at the zero level set.
## This function is irregular, it is thus not recommended for use.

def phi_binary(mask):

    phi_bin = np.ones(mask.shape)
    phi_bin[np.where(mask != 0)] = -1

    return  phi_bin


def display_mesh(verts, faces, normals, texture, save_path):

    mesh = vv.mesh(verts, faces, normals, texture)
    f = vv.gca()
    mesh.colormap = vv.CM_JET
    f.axis.visible = False
    #f.bgcolor = None #1,1,1 #None
    #mesh.edgeShading = 'smooth'
    #mesh.clim = np.min(texture),np.max(texture)
    #mesh.clim = -0.05,0.02
    vv.callLater(1.0, vv.screenshot, save_path, vv.gcf(), sf=2, bg='w')
    vv.colorbar()
    vv.view({'zoom': 0.0053, 'azimuth': -80.0, 'elevation': 5.0})
    #vv.view({'zoom': 0.005, 'azimuth': -80.0, 'elevation': -5.0})
    vv.use().Run()

    return 0

#### Affect texture value to each vertex by averaging neighbrhood information
def texture_mean_avg_interpolation3D(verts, texture):

    X = np.rint(verts[:,0]).astype(int)
    Y = np.rint(verts[:,1]).astype(int)
    Z = np.rint(verts[:,2]).astype(int)

    return (texture[X-1,Y,Z] + texture[X+1,Y,Z] + texture[X,Y-1,Z] + texture[X,Y+1,Z] + texture[X,Y,Z-1] + texture[X,Y,Z+1])/6

#### Affect texture value to each vertex by nearest neighbour interpolation
def texture_nearest_neigh_interpolation3D(verts, texture):

    return texture[np.rint(verts[:,0]).astype(int),np.rint(verts[:,1]).astype(int),np.rint(verts[:,2]).astype(int)]
    #return curv


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--mask', help='3D shape binary mask, as NIFTI file', type=str, required = True)
    parser.add_argument('-m', '--mesh', help='surface mesh generated by FreeSurfer, which corresponds to the mask nifti file, \
    as GIFTI file', type=str, required = True)
    parser.add_argument('-o', '--output', help='output directory', type=str, default = './Gaussian_curvature_results3D')
    parser.add_argument('-dmap', '--dmap', help='distance_map: 0 if Euclidean, 1 if geodesic distance map, and 2 if binary step function', type=int, default = 1)

    args = parser.parse_args()

    # Example of use : python3 Gauss_curv_4_Freesurfer_output.py -in /home/karim/Bureau/Courbure/data/Guillaume_data/rh_white.nii.gz
    #-m /home/karim/Bureau/Courbure/data/Guillaume_data/rh.white.gii

    output_path = args.output

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    nii = nib.load(args.mask)

    shape = nii.get_data()

    affine = nii.affine


    start_time = timeit.default_timer()

    #shape = bbox_3D(shape,5)

    if (args.dmap == 1):

        phi = phi(shape) ## signed geodesic distance

    elif (args.dmap == 2):

        phi = phi_binary(shape) ## binary step function

    else:

        phi = phi_Euclidean(shape) ## signed Euclidean distance


    gaussian_filter(phi, sigma=2, output=phi) ## smoothing of the level set signed distance function

#################### Computation of  Gaussian curvature ###################
    phi_grad, hessian = hessian(phi)
    Ha = hessian_adjoint(hessian)
    Gaussian_curvature = Gaussian_curvature(phi_grad, Ha)
    #Gaussian_curvature = Hessian_adjoint_curvature(phi_grad,Ha)

############################################################################

    elapsed = timeit.default_timer() - start_time
    print("The proposed method takes (in seconds):\n")
    print(elapsed)

    # extract explicitly the implicit surface mesh using the scikit-image toolbox

    mesh = load_mesh(args.mesh)
    ## Express coordinates in the image coordinate system
    verts = map_coordinates(mesh.vertices,np.linalg.inv(affine))
    faces = mesh.faces
    normals = mesh.vertex_normals

    ### Affect per-vertex curvature values, with a nearest neighbour interpolation of vertices on the grid

    #gaussian_curv = texture_nearest_neigh_interpolation3D(verts, Gaussian_curvature)
    gaussian_curv = texture_mean_avg_interpolation3D(verts, Gaussian_curvature)

    print(np.min(gaussian_curv),np.max(gaussian_curv),np.mean(gaussian_curv))

    #### Save results as numpy array

    res = np.append(verts,gaussian_curv[...,None],axis=1)
    np.save(os.path.join(output_path, "gaussian_curv.npy"), res)
    print(res.shape)

    ## Display result

    display_mesh(verts, faces, normals, gaussian_curv, os.path.join(output_path, "Gaussian_curature_Makki.png"))


##To compare results with other methods defining the surface explicitly, please comment/uncomment the following blocks ###############


# #######################################################################################################################################
# ############### To compare results with the Trimesh Gaussian curvature, please uncomment this block ##################################
#
#     start_time = timeit.default_timer()
#
#     #tr_gaussian_curv = curvature.discrete_gaussian_curvature_measure(m, m.vertices, 2)
#     tr_gaussian_curv = curvature.discrete_gaussian_curvature_measure(mesh, verts, 0.5)
#
#     elapsed = timeit.default_timer() - start_time
#
#     print("The Trimesh method takes (in seconds):\n")
#
#     print(elapsed)
#
#     display_mesh(verts, faces, normals, tr_gaussian_curv, os.path.join(output_path, "Gaussian_curvature_Trimesh.png"))
#
# #########################################################################################################################################

# #######################################################################################################################################
# ##### To compare results with the Rusinkiewicz (v1) Gaussian curvature, please uncomment this block ###################################
#
#     start_time = timeit.default_timer()
#     # Comptue estimations of principal curvatures
#     PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(mesh)
#     gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
#
#     elapsed = timeit.default_timer() - start_time
#
#     print("The Rusinkiewicz method v1 takes (in seconds):\n")
#     print(elapsed)
#
#     #gaussian_filter(gaussian_curv, sigma=1, output=gaussian_curv)
#     display_mesh(verts, faces, normals, gaussian_curv, os.path.join(output_path, "Gaussian_curvature_Rusinkiewicz_v1.png"))
# #########################################################################################################################################


# #########################################################################################################################################
# ##### To compare results with the Rusinkiewicz (v2) Gaussian curvature, please uncomment this block #####################################
# ########################### Note that the second version is quite  faster than the first ################################################
#
#     start_time = timeit.default_timer()
#
#     #K,H,VN = WpFcurv.GetCurvatures(m.vertices,m.faces)
#     gaussian_curv = WpFcurv.GetCurvatures(verts,faces)[0]
#
#     elapsed = timeit.default_timer() - start_time
#
#
#     print("The Rusinkiewicz method v2 takes (in seconds):\n")
#     print(elapsed)
#     #print(np.min(gaussian_curv),np.max(gaussian_curv), np.sqrt(np.absolute(np.mean(gaussian_curv)-(1/R**2))))
#
#     #gaussian_filter(gaussian_curv, sigma=1, output=gaussian_curv)
#     display_mesh(verts, faces, normals, gaussian_curv, os.path.join(output_path, "Gaussian_curvature_Rusinkiewicz_v2.png"))
# #########################################################################################################################################


# #########################################################################################################################################
# ##### To compare results with those of the cubic order algorithm, please uncomment this block ###########################################
#
#     start_time = timeit.default_timer()
#
#     #K,H,VN = ccurv.CurvatureCubic(m.vertices,m.faces)
#     gaussian_curv = ccurv.CurvatureCubic(verts,faces)[0]
#
#     elapsed = timeit.default_timer() - start_time
#
#     print("The cubic order algorithm takes (in seconds):\n")
#     print(elapsed)
#
#     #gaussian_filter(gaussian_curv, sigma=1, output=gaussian_curv)
#     display_mesh(verts, faces, normals, gaussian_curv, os.path.join(output_path, "Gaussian_curvature_cubic_order.png"))
# ##########################################################################################################################################

# #########################################################################################################################################
# ##### To compare results with the iterative fitting method, please uncomment this block #################################################
#
#     start_time = timeit.default_timer()
#
#     gaussian_curv = ISFcurv.CurvatureISF2(verts,faces)[0]
#
#     elapsed = timeit.default_timer() - start_time
#
#     print("The iterative fitting method takes (in seconds):\n")
#     print(elapsed)
#
#     #gaussian_filter(gaussian_curv, sigma=1, output=gaussian_curv)
#     display_mesh(verts, faces, normals, gaussian_curv, os.path.join(output_path, "Gaussian_curvature_iterative_fitting.png"))
# ##########################################################################################################################################

# #########################################################################################################################################
# ############## To compare results with the method of Meyer, please uncomment this block #################################################
#
#
#     start_time = timeit.default_timer()
#
#     A_mixed, mean_curvature_normal_operator_vector = diffgeo.calc_A_mixed(verts, faces)
#     gaussian_curv = diffgeo.get_gaussian_curvature(verts, faces, A_mixed)
#
#     elapsed = timeit.default_timer() - start_time
#
#     print("The method of Meyer takes (in seconds):\n")
#     print(elapsed)
#
#     display_mesh(verts, faces, normals, gaussian_curv, os.path.join(output_path, "Gaussian_curvature_Meyer.png"))
# ##########################################################################################################################################
