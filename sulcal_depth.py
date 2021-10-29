# -*- coding: utf-8 -*-
"""
  Author(s): Karim Makki

In this script, we reproduce the method of [1]. However, in [1], the
sulcal depth map was defined as the shortest paths from the 3D convex
hull to the vertices of the cortical surface. And the shortest paths
were computed using the Dijkstra algorithm. A major drawback of the
Dijkstra algorithm is that the direction along which distance increases,
is partially ignored (i.e. only vertical and horizontal displacements are
allowed). One can imagine, for instance, that this algorithm will overestimate
the straight-line Euclidean distance of any diagonal path, crossing a regular grid.
To surmount this issue, we readapted here this method by estimating sulcal depth maps
using the Fast marching method to compute shortest paths.

[1] H. J. Yun, K. Im, J.-J. Yang, U. Yoon, and J.-M. Lee, "Automated sulcal
depth measurement on cortical surface reflecting geometrical properties of
sulci", PLoS ONE, vol. 8, no. 2, Feb. 2013, Art. no. e55977.
"""

import numpy as np
import nibabel as nib
import argparse
import os
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import  skfmm
import visvis as vv
from skimage import measure
import trimesh
import timeit

import fast_Gaussian_curvature_3D as g3D

from curvatures_narrowband import *



def geodesic_distance_to_mask(mask):

    return skfmm.distance(np.subtract(np.max(mask), mask))


def flood_fill_hull(mask):

    points = np.transpose(np.where(mask))
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(mask.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    closed_mask = np.zeros(mask.shape)
    closed_mask[out_idx] = 1

    return closed_mask, hull


def display_mesh(verts, faces, normals, texture, save_path):

    mesh = vv.mesh(verts, faces, normals, texture)
    f = vv.gca()
    #mesh.colormap = vv.CM_WINTER
    mesh.colormap = vv.CM_JET
    f.axis.visible = False
    mesh.clim = 0, 5
    vv.callLater(1.0, vv.screenshot, save_path, vv.gcf(), sf=2, bg='w')
    vv.colorbar()
    #vv.view({'zoom': 0.005, 'azimuth': 80.0, 'elevation': 5.0}) #for the left hemisphere
    vv.view({'zoom': 0.005, 'azimuth': -80.0, 'elevation': -5.0}) # for the right hemisphere
    vv.use().Run()

    return 0

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--anatomical', help='binary mask of the white matter (or right/left hemispheres)', type=str, required = True)
    parser.add_argument('-o', '--output', help='output directory', type=str, default = '../../output_sulcal_depth')

    args = parser.parse_args()


    #### Example of use: python3 sulcal_depth.py -in ./3D_data/Freesurfer_output/rh_white.nii.gz -o /home/karim/Bureau/Courbure/test/sulci_R

    if not os.path.exists(args.output):
        os.makedirs(args.output)


    S = nib.load(args.anatomical).get_fdata()

    S,dx,dy,dz = g3D.bbox_3D(S,15)

    ## Compute convex_hull surface mesh

    convex_hull, hull = flood_fill_hull(S)

    phi = g3D.phi(convex_hull)

    gaussian_filter(phi, sigma=3, output=phi)

    verts, faces, normals, values = measure.marching_cubes_lewiner(phi, 1.)

    ####
    depth = geodesic_distance_to_mask(S)

    texture = g3D.texture_spline_interpolation3D(verts, depth-1)

    verts = g3D.align_origin_back(verts,dx,dy,dz)

    m = trimesh.Trimesh(vertices=verts, faces=faces)
    m.export(os.path.join(args.output, "convex_hull_surface_mesh.obj"))

    display_mesh(verts, faces, normals, texture, os.path.join(args.output, "sulcal_depth_right_hemisphere.png"))
    #display_mesh(verts, faces, normals, texture, os.path.join(args.output, "sulcal_depth_left_hemisphere.png"))
