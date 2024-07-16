import numpy as np

def find_unit_vector_between_two_points(starting_point, terminal_point):
    direction_vector = terminal_point - starting_point
    direction_vector_modulus = np.linalg.norm(direction_vector)
    unit_vector = direction_vector / direction_vector_modulus
    return unit_vector

def find_rotation(coords, vec_o, vec_a):
    vec_from_axis = []
    for i in range(0, coords.shape[1]):
        vec = find_orthoganol_distance_to_axis(coords.T[i], vec_a, vec_o)
        vec_from_axis.append(vec)
    angles = []
    for i in range(0, (len(vec_from_axis)-1)):
        point = vec_from_axis[i]
        corresponding = vec_from_axis[i + 1]
        angle = angle_between(point, corresponding)
        angles.append(angle)
    cumulative_angles = np.cumsum(angles)
    cumulative_angles = np.insert(cumulative_angles, 0, 0)
    return cumulative_angles

def find_orthoganol_distance_to_axis(coords, a, o):
    '''input:
        centroid point coordinates
        vector a - direction vector for cylinder 
        vector o - perpendicular vector from origin to axis
        point_C -  center of cylinder (this is a point on the axis)
       output:
        orthogonal distance from the axis
    '''
    return coords - o - (np.dot(coords, a))*a  # should i just replace o here with ---> point_C + np.dot((origin - point_C), a)*a ## conclusion probably not, need o to be a parameter that is minimised

def angle_between(v1, v2):
    dot_pr = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.arccos(dot_pr / norms)