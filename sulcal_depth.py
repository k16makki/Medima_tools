# -*- coding: utf-8 -*-
"""
  Author(s): Karim Makki
"""

import numpy as np
import nibabel as nib
from numpy import linalg as la
import argparse
import os
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import  skfmm
import visvis as vv
from skimage import measure
import trimesh
import timeit

import fast_Gaussian_curvature_3D as g3D

def distance_to_mask(mask):

    d = np.subtract(np.max(mask), mask)
    #return ndimage.distance_transform_edt(d)
    return skfmm.distance(d)


def flood_fill_hull(mask):

    points = np.transpose(np.where(mask))
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(mask.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    closed_mask = np.zeros(mask.shape)
    closed_mask[out_idx] = 1

    return closed_mask, hull


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--anatomical', help='binary mask of the white matter (or right/left hemispheres)', type=str, required = True)
    parser.add_argument('-o', '--output', help='output directory', type=str, default = './output_sulcal_depth')

    args = parser.parse_args()


    #### Example of use: python3 sulcal_depth.py -in /home/karim/Bureau/Courbure/data/subject1-session1_R_white.nii.gz -o /home/karim/Bureau/Courbure/test/sulci_R

    if not os.path.exists(args.output):
        os.makedirs(args.output)


    S = nib.load(args.anatomical).get_fdata()

    S = g3D.bbox_3D(S,5)



    start_time = timeit.default_timer()

    out_boundary, hull = flood_fill_hull(S)
    S_prime = np.max(out_boundary) - out_boundary
    distance_map = distance_to_mask(out_boundary)
    S_prime[np.where(distance_map<=1)]= 0

    r = np.zeros(S.shape)

    R = np.where(S+S_prime == 0)

    r[R] = 1

    print(len(R[0]))

    nx, ny, nz = S.shape #Number of steps in space(x), (y) and (z)

    r[...,0] = 0
    r[...,nz-1] = 0
    r[0,...] = 0
    r[nx-1,...] = 0
    r[:,0,:] = 0
    r[:,ny-1,:] = 0

	#Initial conditions

    hdr = nib.load(args.anatomical).header # Anatomical image header: this will give the voxel spacing along each axis (dx, dy, dz)
    dx = hdr.get_zooms()[0] #x-voxel spacing
    dy = hdr.get_zooms()[1] #y-voxel spacing
    dz = hdr.get_zooms()[2] #z-voxel spacing

    voxel_volume = dx * dy * dz

    u = np.zeros((nx,ny,nz))  #u(i)
    u_n = np.zeros((nx,ny,nz)) #u(i+1)

    L0 = np.zeros((nx,ny,nz))
    L1 = np.zeros((nx,ny,nz))


    L0_n = np.zeros((nx,ny,nz))
    L1_n = np.zeros((nx,ny,nz))

    Nx = np.zeros((nx,ny,nz))
    Ny = np.zeros((nx,ny,nz))
    Nz = np.zeros((nx,ny,nz))


    depth = np.zeros((nx,ny,nz))

	#Dirichlet boundary conditions
    u_max = 100

    u[np.where(S_prime != 0)] =  u_max

	# Boundary conditions for computing L0 and L1

    L0[:,:,:]= -(dx+dy+dz)/6
    L1[:,:,:]= -(dx+dy+dz)/6

	# Explicit iterative scheme to solve the Laplace equation by the simplest method  (the  Jacobi method).
	# We obtain the harmonic function u(x,y,z) by solving the Laplace equation over R

    n_iter = 200

    for it in range (n_iter):

        u_n = u

        u[R[0],R[1],R[2]]= (u_n[R[0]+1,R[1],R[2]] + u_n[R[0]-1,R[1],R[2]] +  u_n[R[0],R[1]+1,R[2]] \
        + u_n[R[0],R[1]-1,R[2]] + u_n[R[0],R[1],R[2]+1] + u_n[R[0],R[1],R[2]-1]) / 6



    print("Laplace's equation is solved")
    del u_n

	##Compute the normalized tangent vector field of the correspondence trajectories

    N_xx, N_yy, N_zz = np.gradient(u)

    grad_norm = np.sqrt(N_xx**2 + N_yy**2 + N_zz**2)

    grad_norm[np.where(grad_norm==0)] = 1 ## to avoid dividing by zero

	# Normalization
    np.divide(N_xx, grad_norm, Nx)
    np.divide(N_yy, grad_norm, Ny)
    np.divide(N_zz, grad_norm, Nz)

    gaussian_filter(Nx, sigma=2, output=Nx)
    gaussian_filter(Ny, sigma=2, output=Ny)
    gaussian_filter(Nz, sigma=2, output=Nz)

    del grad_norm, N_xx, N_yy, N_zz

    print("The normalized tangent vector field is successfully computed")

    den = np.absolute(Nx)+ np.absolute(Ny) + np.absolute(Nz)

    den[np.where(den==0)] = 1 ## to avoid dividing by zero

	# iteratively compute correspondence trajectory lengths L0 and L1

    for it in range (100):

        L0_n = L0
        L1_n = L1

        L0[R[0],R[1],R[2]] =  (1 + np.absolute(Nx[R[0],R[1],R[2]])* L0_n[(R[0]-np.sign(Nx[R[0],R[1],R[2]])).astype(int),R[1],R[2]]+ \
        np.absolute(Ny[R[0],R[1],R[2]]) * L0_n[R[0],(R[1]-np.sign(Ny[R[0],R[1],R[2]])).astype(int),R[2]]  \
        + np.absolute(Nz[R[0],R[1],R[2]]) * L0_n[R[0],R[1],(R[2]-np.sign(Nz[R[0],R[1],R[2]])).astype(int)])  /   den[R[0],R[1],R[2]]

        L1[R[0],R[1],R[2]] =  (1 + np.absolute(Nx[R[0],R[1],R[2]])* L1_n[(R[0]+np.sign(Nx[R[0],R[1],R[2]])).astype(int),R[1],R[2]]+ \
        np.absolute(Ny[R[0],R[1],R[2]]) * L1_n[R[0],(R[1]+np.sign(Ny[R[0],R[1],R[2]])).astype(int),R[2]]  \
        + np.absolute(Nz[R[0],R[1],R[2]]) * L1_n[R[0],R[1],(R[2]+np.sign(Nz[R[0],R[1],R[2]])).astype(int)])  /   den[R[0],R[1],R[2]]



	# compute  the thickness of the tissue region inside R
    del L0_n, L1_n

    depth[R[0],R[1],R[2]] = L0[R[0],R[1],R[2]] + L1[R[0],R[1],R[2]]

    depth[R[0],R[1],R[2]] -= np.min(depth[R[0],R[1],R[2]])

    elapsed = timeit.default_timer() - start_time
    print("Sulcal depth is successfully computed within (time in seconds):\n")

    print(elapsed)



    print("Mean sulcal depth:\n")
    print(np.mean(depth[R[0],R[1],R[2]]))
    print("Maximum sulcal depth:\n")
    print(np.max(depth[R[0],R[1],R[2]]))
    print("Minimum sulcal depth:\n")
    print(np.min(depth[R[0],R[1],R[2]]))

    ## To reduce the effects of voxelisation when computing the finite differences, we convolve the result with a 3D Gaussian filter

    #gaussian_filter(depth, sigma=(0.5, 0.5, 0.5), output=depth)



    # Save results as 3D nifti files

    nii = nib.load(args.anatomical)
    i = nib.Nifti1Image(u, nii.affine)
    j = nib.Nifti1Image(depth, nii.affine)
    k = nib.Nifti1Image(L0, nii.affine)
    l = nib.Nifti1Image(L1, nii.affine)

    nib.save(i, args.output + '/harmonic_function.nii.gz')
    nib.save(j, args.output + '/sulcal_depth.nii.gz')
    nib.save(k, args.output + '/L0.nii.gz')
    nib.save(l, args.output + '/L1.nii.gz')

    gaussian_filter(u, sigma=2, output=u)
    verts, faces, normals, values = measure.marching_cubes_lewiner(u, 1) ## surface mesh
    #verts, faces, normals, values = measure.marching_cubes_lewiner(u, u_max-5) ##  To display results on convex surface
    m = trimesh.Trimesh(vertices=verts, faces=faces)
    m.export(args.output+'/dilated_surface_mesh.ply')


    texture = depth[verts[:,0].astype(int),verts[:,1].astype(int),verts[:,2].astype(int)]
    print(np.min(texture),np.max(texture))
    mesh = vv.mesh(verts, faces, normals, texture)

    f = vv.gca()
    mesh.colormap = vv.CM_HOT
    vv.callLater(1.0, vv.screenshot, args.output + '/sulcal_depth.png', vv.gcf(), sf=2, bg='w')
    vv.colorbar()
    #vv.view({'zoom': 0.0053, 'azimuth': -65.0, 'elevation': 65.0})
    f.axis.visible = False
    vv.view({'zoom': 0.0053, 'azimuth': -80.0, 'elevation': 5.0})
    vv.use().Run()
