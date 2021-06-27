##Source: https://github.com/justachetan/DiffGeoOps

#This script represents a Python implementation of the paper:
#Discrete Differential-Geometry Operators for Triangulated 2-Manifolds. Mark Meyer, Mathieu Desbrun, Peter Schröder and Alan H. Barr. VisMath 2002



import numpy as np



def get_heron_area(a, b, c):

    x = np.linalg.norm((b - a), 2)
    y = np.linalg.norm((c - a), 2)
    z = np.linalg.norm((c - b), 2)
    s = (x + y + z) * 0.5

    return (s * (s - x) * (s - y) * (s - z)) ** 0.5


def calc_A_mixed(vertices, triangles):

    numv = vertices.shape[0]
    numt = triangles.shape[0]

    A_mixed = np.zeros((numv, numt), dtype='float16')

    mean_curvature_normal_operator = np.zeros((numv, numt, 3), dtype='float16')

    for i in range(numv):

        req_t = triangles[(triangles[:, 0] == i) | (
            triangles[:, 1] == i) | (triangles[:, 2] == i)]

        for j in range(len(req_t)):

            tid = np.where(np.all(triangles == req_t[j], axis=1))

            nbhr = [v for v in req_t[j] if v != i]

            vec1 = (vertices[nbhr[0]] - vertices[i]) / \
                np.linalg.norm(vertices[nbhr[0]] - vertices[i], 2)
            vec2 = (vertices[nbhr[1]] - vertices[i]) / \
                np.linalg.norm(vertices[nbhr[1]] - vertices[i], 2)
            angle_at_x = np.arccos(np.dot(vec1, vec2))

            if angle_at_x > np.pi / 2:
                A_mixed[i, tid] = get_heron_area(
                    vertices[i], vertices[nbhr[0]], vertices[nbhr[1]]) / 2
                continue

            vec1a = (vertices[i] - vertices[nbhr[0]]) / \
                np.linalg.norm(vertices[i] - vertices[nbhr[0]], 2)
            vec2a = (vertices[nbhr[1]] - vertices[nbhr[0]]) / \
                np.linalg.norm(vertices[nbhr[1]] - vertices[nbhr[0]], 2)

            inner_prod = np.dot(vec1a, vec2a)
            angle1 = np.arccos(inner_prod)

            if angle1 > np.pi / 2:
                A_mixed[i, tid] = get_heron_area(
                    vertices[i], vertices[nbhr[0]], vertices[nbhr[1]]) / 4
                continue

            vec1b = (vertices[i] - vertices[nbhr[1]]) / \
                np.linalg.norm(vertices[i] - vertices[nbhr[1]], 2)
            vec2b = (vertices[nbhr[0]] - vertices[nbhr[1]]) / \
                np.linalg.norm(vertices[nbhr[0]] - vertices[nbhr[1]], 2)

            inner_prod = np.dot(vec1b, vec2b)
            angle2 = np.arccos(inner_prod)

            if angle2 > np.pi / 2:
                A_mixed[i, tid] = get_heron_area(
                    vertices[i], vertices[nbhr[0]], vertices[nbhr[1]]) / 4
                continue

            cot_1 = 1 / np.tan(angle1)
            cot_2 = 1 / np.tan(angle2)

            A_v_of_tid = 0.125 * ((cot_1 * np.linalg.norm(vertices[i] - vertices[nbhr[
                1]], 2)**2) + (cot_2 * np.linalg.norm(vertices[i] - vertices[nbhr[0]], 2)**2))

            mean_curvature_normal_operator_at_v_t = ((1 / np.tan(angle1)) * (
                vertices[i] - vertices[nbhr[1]])) + ((1 / np.tan(angle2)) * (vertices[i] - vertices[nbhr[0]]))

            A_mixed[i, tid] = A_v_of_tid
            mean_curvature_normal_operator[
                i, tid] = mean_curvature_normal_operator_at_v_t

    A_mixed = np.sum(A_mixed, axis=1)
    # Set zeros in A_mixed to very small values
    A_mixed[A_mixed == 0] = 10 ** -40
    mean_curvature_normal_operator = (
        (1 / (2 * A_mixed)) * np.sum(mean_curvature_normal_operator, axis=1).T).T

    return A_mixed, mean_curvature_normal_operator


def get_mean_curvature(mean_curvature_normal_operator_vector):
    K_H = 0.5 * \
        np.linalg.norm(mean_curvature_normal_operator_vector, 2, axis=1)
    return K_H


def get_gaussian_curvature(vertices, triangles, A_mixed):
    numv = vertices.shape[0]
    numt = triangles.shape[0]
    K_G = np.zeros(numv)
    for i in range(numv):
        sum_theta = 0
        req_t = triangles[(triangles[:, 0] == i) | (
            triangles[:, 1] == i) | (triangles[:, 2] == i)]

        for j in range(req_t.shape[0]):

            nbhrs = [v for v in req_t[j] if v != i]
            vec1 = vertices[nbhrs[0]] - vertices[i]
            vec1 = vec1 / np.linalg.norm(vec1, 2)
            vec2 = vertices[nbhrs[1]] - vertices[i]
            vec2 = vec2 / np.linalg.norm(vec2, 2)
            angle = np.arccos(np.dot(vec1, vec2))
            sum_theta += angle

        K_G[i] = ((2 * np.pi) - sum_theta) / A_mixed[i]
    return K_G


def get_principal_curvatures(K_H, K_G):
    numv = vertices.shape[0]
    numt = triangles.shape[0]
    zeros = np.zeros(numv)
    delx = np.sqrt(np.max(np.vstack((K_H**2 - K_G, zeros)), axis=0))
    K_1 = K_H + delx
    K_2 = K_H - delx
    return K_1, K_2
