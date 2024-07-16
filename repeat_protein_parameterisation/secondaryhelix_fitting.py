import numpy as np
from scipy.optimize import fminbound
from utils import find_unit_vector_between_two_points, find_rotation, angle_between

def finding_vectorV_W_start(vec_o, vec_a, term_point):
    # Find the estimated starting point of the helix
    starting_point = vec_o + np.dot((term_point - vec_o), vec_a)*vec_a
    # Find vector v (starting point of the helix to the first data point)
    vector_v = find_unit_vector_between_two_points(starting_point, term_point)
    # vector w is perp to both a and v, thus the cross produce of vec_a and vec_v should result in a vector they are BOTH perpendicular to
    vector_w = np.cross(vec_a, vector_v)
    # Rotation around the axis
    return starting_point, vector_v, vector_w

def finding_t_arr(refined_est, handedness, Ca_coords, centroids):
    pitch, radius, ax, ay, az, ox, oy, oz, tx, ty, tz = refined_est
    vec_a = np.array([ax, ay, az])
    vec_o = np.array([ox, oy, oz])
    term_point = np.array([tx, ty, tz])
    # finding t which is able to be varied
    full_rotation = find_rotation(centroids, vec_o, vec_a)
    
    if Ca_coords.shape[1] > centroids.shape[1]:
        mean = full_rotation[-1]/centroids.shape[1]
        new_value = full_rotation[-1] + mean
        full_rotation = np.insert(full_rotation, -1, new_value)
    
    if handedness == "left":  # is this the correct assingment or is it left twist that requires this?
        m = -1
    else:
        m = 1
    starting_point, vector_v, vector_w = finding_vectorV_W_start(vec_o, vec_a, term_point)
    # find the value of t which produces the minimum distance
    t_list = []
    for i in range(0, Ca_coords.shape[1]):
        x = Ca_coords.T[i][0]
        y = Ca_coords.T[i][1]
        z = Ca_coords.T[i][2]
        pmin = full_rotation[i] - (np.pi/2)
        pmax = full_rotation[i] + (np.pi/2)
        def g(t): return starting_point + (vec_a*pitch*t/(np.pi*2)) + radius*(vector_v*np.cos(t) + vector_w*np.sin(t*m))
        def ggg(t, x, y, z): return np.sum((g(t)-[x, y, z])**2)
        bestFitValue = fminbound(ggg, pmin, pmax, args=(x, y, z))
        best = bestFitValue
        t_list.append(best)
    t_list = np.array(t_list)
    return t_list

def helix_reconstruct_2(t_list, refined_est, handedness):
    pitch, radius, ax, ay, az, ox, oy, oz, tx, ty, tz = refined_est
    vec_a = np.array([ax, ay, az])
    vec_o = np.array([ox, oy, oz])
    term_point = np.array([tx, ty, tz])
    starting_point, vector_v, vector_w = finding_vectorV_W_start(
        vec_o, vec_a, term_point)
    if handedness == "left":  # is this the correct assingment or is it left twist that requires this?
        m = -1
    else:
        m = 1
    helix_coords = []
    for i in t_list:
        helix_point = starting_point + (vec_a*pitch*i/(np.pi*2)) + radius*(vector_v*np.cos(i) + vector_w*np.sin(i*m))  # sin * m
        helix_coords.append(list(helix_point))
    helix_coords = np.array(helix_coords)
    return helix_coords.T


def estimate_radius_alpha(Ca_coords, test_helix):
    distance = []
    for i in range(0, len(Ca_coords)):
        dist = np.linalg.norm(Ca_coords[i] - test_helix.T[i])
        distance.append(dist)
    distance_sum = np.sum(distance, axis=0)
    average_distance = distance_sum / len(distance)
    return average_distance


def frenent_frame(helix):
    # Calculate the first and second derivative of the points
    dX = np.apply_along_axis(np.gradient, axis=0, arr=helix.T)
    ddX = np.apply_along_axis(np.gradient, axis=0, arr=dX)
    # Normalize all tangents, find the unit tangent vector (T)
    def f(m): return m / np.linalg.norm(m)
    T = np.apply_along_axis(f, axis=1, arr=dX)
    # Calculate and normalize all binormals, find the binormal vector (B)
    B = np.cross(dX, ddX)
    B = np.apply_along_axis(f, axis=1, arr=B)
    # Calculate all normals, find the unit normal vector (N)
    N = np.cross(B, T)
    return T, B, N


def helix_3(estimates, Ca_coords, helix, t_vals, B, N):
    omega, alpha, phase = estimates
    helix_coordinates = []
    for i in range(0, Ca_coords.shape[1]):
        theta = t_vals[i]*omega + phase
        X = helix.T[i] + alpha*np.cos(theta)*N[i] + alpha*np.sin(theta)*B[i]
        helix_coordinates.append(list(X))
    double_helix = np.array(helix_coordinates)
    return double_helix


def cost_function_3(estimates, Ca_coords, helix, t_vals, B, N):
    '''input:
        centroid point coordinates
        vector a - direction vector for cylinder 
        vector o - perpendicular vector from origin to axis
        r - radius
       output:
        sum of squared distances (i.e the output of the cost function and what we intend to minimise to fit this cylinder)
    '''
    omega, alpha, phase = estimates

    dist = []
    for i in range(0, Ca_coords.T.shape[1]):
        dist_indiv = (Ca_coords[i] - helix_3(estimates, Ca_coords.T, helix, t_vals, B, N)[i])**2
        dist.append(dist_indiv)
    res = np.sum(dist)
    return res

def estimate_phase(helix, Ca_res):
    T, B, N = frenent_frame(helix)
    phase_arr = []
    for i in range(0, helix.shape[1]):
        ## find the angle between two vectors
        A = helix.T[i]
        # 1) first vector from centroid to Ca
        P = Ca_res[i]
        AP = A - P
        # 2) vector from centorid to rough first helix point
        Q = A + np.cos(0)*N[i] + np.sin(0)*B[i]  # point at 0 phase
        AQ = A - Q
        # find the normal to the plane defined by these two vectors
        normal = np.cross(AP, AQ)
        #make sure that the normal is in the direction of the tangent to that point
        if np.dot(normal, T[0]) < 0:
            normal = normal * -1
        else:
            pass
        # find the angle between two vectors
        phase = angle_between(AP, AQ)
        omega = np.linalg.det(np.dstack([AP, AQ, normal]))
        if omega > 0:
            phase = np.pi*2 - phase
        phase_arr.append(phase)
    phase_first = phase_arr[0]
    return phase_first