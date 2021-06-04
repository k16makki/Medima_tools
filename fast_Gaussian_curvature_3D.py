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

    np.divide(gaussian_curv,np.power(L2_norm_grad(gx,gy,gz),4),gaussian_curv)
    gaussian_filter(gaussian_curv, sigma=1, output=gaussian_curv)

    return gaussian_curv


def bbox_3D(mask,depth):

    x = np.any(mask, axis=(1, 2))
    y = np.any(mask, axis=(0, 2))
    z = np.any(mask, axis=(0, 1))
    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return mask[xmin-depth:xmax+depth,ymin-depth:ymax+depth,zmin-depth:zmax+depth]


## signed geodesic distance function for the implicit surface
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


def display_mesh(verts, faces, normals, texture, save_path):

    mesh = vv.mesh(verts, faces, normals, texture)#, verticesPerFace=3)
    f = vv.gca()
    mesh.colormap = vv.CM_JET
    #mesh.edgeShading = 'smooth'
    #mesh.clim = np.min(texture),np.max(texture)
    vv.callLater(1.0, vv.screenshot, save_path, vv.gcf(), sf=2)
    vv.colorbar()
    #vv.view({'azimuth': 45.0, 'elevation': 45.0})
    f.axis.visible = False
    vv.use().Run()

    return 0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--mask', help='3D shape binary mask, as NIFTI file', type=str, required = True)
    parser.add_argument('-o', '--output', help='output directory', type=str, default = './Gaussian_curvature_results')
    parser.add_argument('-dmap', '--dmap', help='distance_map: 0 if Euclidean, 1 if geodesic distance map, and 2 if binary step function', type=int, default = 0)

    args = parser.parse_args()

    # Example of use : python3 fast_Gaussian_curvature_3D.py -in ./3D_data/stanford_bunny_binary.nii.gz

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

#################### Computation of  Gaussian curvature ###################
    phi_grad, hessian = hessian(phi)
    Ha = hessian_adjoint(hessian)
    Gaussian_curvature = Gaussian_curvature(phi_grad, Ha)

############################################################################

    elapsed = timeit.default_timer() - start_time
    print("The proposed method takes (in seconds):\n")
    print(elapsed)

    # extract explicitly the implicit surface mesh using the scikit-image toolbox

    #hdr = nib.load(args.mask).header
    #dx = hdr.get_zooms()[0] #x-voxel spacing
    #dy = hdr.get_zooms()[1] #y-voxel spacing
    #dz = hdr.get_zooms()[2] #z-voxel spacing

    verts, faces, normals, values = measure.marching_cubes_lewiner(phi, 0.0)#, spacing=(dx,dy,dz), gradient_direction='descent')
    print(verts.shape)

    m = trimesh.Trimesh(vertices=verts, faces=faces)

    m.export(os.path.join(output_path, "surface_mesh.obj"))

    texture = Gaussian_curvature[verts[:,0].astype(int),verts[:,1].astype(int),verts[:,2].astype(int)]
    display_mesh(verts, faces, normals, texture, os.path.join(output_path, "Gaussian_curature_Makki.png"))

####To compare results with other methods defining the surface explicitly, please comment/uncomment the following blocks ###############

# #######################################################################################################################################
# ##### To compare results with the Rusinkiewicz (v1) Gaussian curvature, please uncomment the following block ##########################
#
#     m = trimesh.load_mesh(os.path.join(output_path, "surface_mesh.obj"))
#
#     start_time = timeit.default_timer()
#     # Comptue estimations of principal curvatures
#     PrincipalCurvatures, PrincipalDir1, PrincipalDir2 = scurv.curvatures_and_derivatives(m)
#     gaussian_curv = PrincipalCurvatures[0, :] * PrincipalCurvatures[1, :]
#
#     elapsed = timeit.default_timer() - start_time
#
#     print("The Rusinkiewicz method v1 takes (in seconds):\n")
#     print(elapsed)
#
#     #gaussian_filter(gaussian_curv, sigma=1, output=gaussian_curv)
#     display_mesh(m.vertices, m.faces, m.vertex_normals, gaussian_curv, os.path.join(output_path, "Gaussian_curvature_Rusinkiewicz_v1.png"))
# #########################################################################################################################################


#########################################################################################################################################
##### To compare results with the Rusinkiewicz (v2) Gaussian curvature, please uncomment the following block ############################
########################### Note that the second version is quite  faster than the first ################################################

    m = trimesh.load_mesh(os.path.join(output_path, "surface_mesh.obj"))

    start_time = timeit.default_timer()

    #K,H,VN = WpFcurv.GetCurvatures(m.vertices,m.faces)
    gaussian_curv = WpFcurv.GetCurvatures(m.vertices,m.faces)[0]

    elapsed = timeit.default_timer() - start_time

    print("The Rusinkiewicz method v2 takes (in seconds):\n")
    print(elapsed)

    #gaussian_filter(gaussian_curv, sigma=1, output=gaussian_curv)
    display_mesh(m.vertices, m.faces, m.vertex_normals, gaussian_curv, os.path.join(output_path, "Gaussian_curvature_Rusinkiewicz_v2.png"))
##########################################################################################################################################


# #########################################################################################################################################
# ##### To compare results with those of the cubic order algorithm, please uncomment the following block ##################################
#
#     m = trimesh.load_mesh(os.path.join(output_path, "surface_mesh.obj"))
#
#     start_time = timeit.default_timer()
#
#     #K,H,VN = ccurv.CurvatureCubic(m.vertices,m.faces)
#     gaussian_curv = ccurv.CurvatureCubic(m.vertices,m.faces)[0]
#
#     elapsed = timeit.default_timer() - start_time
#
#     print("The cubic order algorithm takes (in seconds):\n")
#     print(elapsed)
#
#     #gaussian_filter(gaussian_curv, sigma=1, output=gaussian_curv)
#     display_mesh(m.vertices, m.faces, m.vertex_normals, gaussian_curv, os.path.join(output_path, "Gaussian_curvature_cubic_order.png"))
# ##########################################################################################################################################
