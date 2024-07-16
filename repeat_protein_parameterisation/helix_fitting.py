import numpy as np
from utils import find_unit_vector_between_two_points, find_rotation, angle_between

def direction(theta, phi):
    '''Return the direction vector of a cylinder defined
    by the spherical coordinates theta and phi.
    '''
    return np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta),
                     np.cos(theta)])


def projection_matrix(w):
    '''Return the projection matrix  of a direction w.'''
    return np.identity(3) - np.dot(np.reshape(w, (3, 1)), np.reshape(w, (1, 3)))


def skew_matrix(w):
    '''Return the skew matrix of a direction w.'''
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])


def calc_A(Ys):
    '''Return the matrix A from a list of Y vectors.'''
    return sum(np.dot(np.reshape(Y, (3, 1)), np.reshape(Y, (1, 3)))
               for Y in Ys)


def calc_A_hat(A, S):
    '''Return the A_hat matrix of A given the skew matrix S'''
    return np.dot(S, np.dot(A, np.transpose(S)))


def preprocess_data(Xs_raw):
    '''Translate the center of mass (COM) of the data to the origin.
    Return the prossed data and the shift of the COM'''
    n = len(Xs_raw)
    Xs_raw_mean = sum(X for X in Xs_raw) / n

    return [X - Xs_raw_mean for X in Xs_raw], Xs_raw_mean


def G(w, Xs):
    '''Calculate the G function given a cylinder direction w and a
    list of data points Xs to be fitted.'''
    n = len(Xs)
    P = projection_matrix(w)
    Ys = [np.dot(P, X) for X in Xs]
    A = calc_A(Ys)
    A_hat = calc_A_hat(A, skew_matrix(w))
    u = sum(np.dot(Y, Y) for Y in Ys) / n
    v = np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / \
        np.trace(np.dot(A_hat, A))
    return sum((np.dot(Y, Y) - u - 2 * np.dot(Y, v)) ** 2 for Y in Ys)


def C(w, Xs):
    '''Calculate the cylinder center given the cylinder direction and 
    a list of data points.
    '''
    P = projection_matrix(w)
    Ys = [np.dot(P, X) for X in Xs]
    A = calc_A(Ys)
    A_hat = calc_A_hat(A, skew_matrix(w))

    return np.dot(A_hat, sum(np.dot(Y, Y) * Y for Y in Ys)) / np.trace(np.dot(A_hat, A))


def est_rad(w, Xs):
    '''Calculate the radius given the cylinder direction and a list
    of data points.
    '''
    n = len(Xs)
    P = projection_matrix(w)
    c = C(w, Xs)
    return np.sqrt(sum(np.dot(c - X, np.dot(P, c - X)) for X in Xs) / n)


def inital_axis_direction(coords):
    # find vectors and normalise them (we will first do this for the first 4 points - may expand and average this later)
    # maybe write a for loop for each set later on and then average the bisector that you get??
    bisector_array, final_arr = [], []
    for i in range(1, (len(coords)-1)):
        C = coords[i-1] - coords[i]
        D = coords[i+1] - coords[i]
        bisector = C*np.linalg.norm(D) + D*np.linalg.norm(C)
        bisector_array.append(list(bisector))
    for i in bisector_array:
        # interact with each element in the array within one dimension
        for j in range(len(bisector_array)):
            if j > bisector_array.index(i):
                axis_vector = np.cross(i, bisector_array[j])
                inital_axis_bisector = axis_vector / \
                    np.linalg.norm(axis_vector)
                final_arr.append(list(inital_axis_bisector))
            else:
                pass
    total_list = np.array(final_arr)
    inital_axis_new = (np.sum(total_list, axis=0))
    inital_axis_average_new = inital_axis_new / np.linalg.norm(inital_axis_new)
    return inital_axis_average_new

# ----------------------------------------
# minimise superhelix parameters - functions
# ----------------------------------------
def find_minimal_perpendicular_vector(a, point_C):
    '''input:
        vector a - direction vector for cylinder 
        point_C -  center of cylinder (this is a point on the axis)
       output:
        orthogonal vector to axis from origin 
    '''
    origin = np.zeros(3,)
    return point_C + np.dot((origin - point_C), a)*a

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

def cost_function(estimates, coords):
    '''input:
        centroid point coordinates
        vector a - direction vector for cylinder 
        vector o - perpendicular vector from origin to axis
        r - radius
       output:
        sum of squared distances (i.e the output of the cost function and what we intend to minimise to fit this cylinder)
    '''
    ax, ay, az, ox, oy, oz, r = estimates
    a = np.array([ax, ay, az])
    o = np.array([ox, oy, oz])
    dist = []
    for i in coords:
        dist_indiv = (np.linalg.norm(find_orthoganol_distance_to_axis(i, a, o)) - r)**2
        dist.append(dist_indiv)
    res = np.sum(dist)
    return res


def constraint_1(estimate):
    ax, ay, az, ox, oy, oz, r = estimate
    a = np.array([ax, ay, az])
    o = np.array([ox, oy, oz])
    return np.dot(a, o)


def constraint_2(estimate):
    ax, ay, az, ox, oy, oz, r = estimate
    a = np.array([ax, ay, az])
    return np.linalg.norm(a) - 1

# ----------------------------------------
# translation and rotation - functions
# ----------------------------------------
def find_minimal_perpendicular_vector_edit_tobe_general(dir_vec, point_line, centroids):
    '''input:
         vector a - direction vector for cylinder 
         point_C -  center of cylinder (this is a point on the axis)
        output:
         orthogonal vector to axis from origin 
     '''
    first_centroid = centroids[0]
    return point_line + np.dot((first_centroid - point_line), dir_vec)*dir_vec


def translate_to_origin(xyz_axis, xyz_coords, new_Ca_coords, starting_point):
    # note -  changed from midpoint -  is this the correct decision???
    central_axis = xyz_axis - starting_point
    translated_solenoid_Cas = xyz_coords - starting_point
    Ca_backbone = new_Ca_coords - starting_point
    return central_axis, translated_solenoid_Cas, Ca_backbone


def rotate_to_z(v, first_z_coord):
    """
    Given a position column vector v in Cartesian space, find rotation angles
    alpha and beta
    that rotates the given vector to z-axis.

    Rotation order matters
    order: (str) if 'xy',
    """
    a, b, c = v.flatten()
    d = {'alpha': None, 'beta': None}
    d['alpha'] = np.arctan(b / c)
    d['beta'] = np.arctan(-a / np.sqrt(b ** 2 + c ** 2))
    if c < 0:
        # if direction vector z coord = negative = pi - beta for inverse rotation
        d['beta'] = np.pi - d['beta']
    else:
        # if direction vector z coord = positive = normal beta for inverse rotation
        pass
    return d['alpha'], d['beta']


def rotMat(axis_, theta):
    """
    Return rotation matrix that rotates a vector around the specified axis
    'axis_' by an angle 'theta' clockwise relative to the axis direction.

    axis_: (str) either, 'x', 'y' or 'z'
    theta: (float) unit radian
    unit : (str) specify the angle units, default 'rad' i.e. radians, if 'deg', then converts the angle to radians
    """
    axis_ = axis_.lower()
    if axis_ == 'x':
        return np.array([[1, 0, 0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta), np.cos(theta)]])
    elif axis_ == 'y':
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])
    elif axis_ == 'z':
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])


def rotation_matrix_from_axis_and_angle(u, theta):
    '''Calculate a rotation matrix from an axis and an angle.
    '''
    x = u[0]
    y = u[1]
    z = u[2]
    s = np.sin(theta)
    c = np.cos(theta)
    return np.array([[c + x**2 * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
                     [y * x * (1 - c) + z * s, c + y**2 *
                      (1 - c), y * z * (1 - c) - x * s],
                     [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z**2 * (1 - c)]])


def make_cylinder_surface(dir_vec, min_val, max_val, radius):
    phi = np.arctan2(dir_vec[1], dir_vec[0])
    M = np.dot(rotation_matrix_from_axis_and_angle(np.array([0, 0, 1]), phi),
               rotation_matrix_from_axis_and_angle(np.array([0, 1, 0]), 0))
    delta = np.linspace(-np.pi, np.pi, 20)
    z = np.linspace(min_val, max_val, 20)
    Delta, Z = np.meshgrid(delta, z)
    X = radius * np.cos(Delta)
    Y = radius * np.sin(Delta)
    for i in range(len(X)):
        for j in range(len(X[i])):
            p = np.dot(M, np.array([X[i][j], Y[i][j], Z[i][j]]))
            X[i][j] = p[0]
            Y[i][j] = p[1]
            Z[i][j] = p[2]
    return X, Y, Z

# ----------------------------------------
# pitch and handedness estimation - functions
# ----------------------------------------
def flatten_coordinates(coords):
    xy_coords = coords[:2]
    z_coords = coords[2]
    return xy_coords, z_coords

def displacement_along_axis(z_coords):
    # Zero coordinates (this should be VERY close just python rounding error)
    normalised_z_coords = z_coords - z_coords[0]
    # find difference between elements
    z_displacement = [
        j-i for i, j in zip(normalised_z_coords[:-1], normalised_z_coords[1:])]
    # sum elements
    cumulative_displacement = np.cumsum(z_displacement)
    return cumulative_displacement

def finding_angles(coords):
    angles = []
    for i in range(coords.shape[1] - 1):
        point = coords[:, i]
        corresponding = coords[:, i + 1]
        ang1 = np.arctan2(*point[::-1])
        ang2 = np.arctan2(*corresponding[::-1])
        angle = (ang1 - ang2) % (2 * np.pi)
        angles.append(angle)
    mean = np.mean(angles)
    if mean < np.pi:
        handedness = "left"
 #       print('--- Left-handed twist: {:.3f} degrees'.format(np.degrees(mean)))
        cumulative_angles = np.cumsum(angles)
    if mean > np.pi:
        handedness = "right"
 #       print('--- Right-handed twist: {:.3f} degrees'.format(360 - np.degrees(mean)))
        new_angles = [abs(angle - 2*np.pi) for angle in angles]
        cumulative_angles = np.cumsum(new_angles)
    return cumulative_angles, handedness

def quadratic_loss_function(rotation, displacement):
    rotation_mean = np.mean(rotation)
    displacement_mean = np.mean(displacement)
    num, den = 0, 0
    for i in range(len(rotation)):
        num += (rotation[i] - rotation_mean) * (displacement[i] - displacement_mean)
        den += (rotation[i] - rotation_mean)**2
    m = num/den
    return m

# ----------------------------------------
# full minimisation - functions
# ----------------------------------------

def helix_reconstruction(estimates, handedness, coords):
    # unpack estimates
    pitch, radius, ax, ay, az, ox, oy, oz, tx, ty, tz = estimates
    vector_a = np.array([ax, ay, az])
    vector_o = np.array([ox, oy, oz])
    terminal_point = np.array([tx, ty, tz])
    # Find the estimated starting point of the helix
    starting_point = vector_o + np.dot((terminal_point - vector_o), vector_a)*vector_a
    # Find vector v (starting point of the helix to the first data point)
    vector_v = find_unit_vector_between_two_points(starting_point, terminal_point)
    # vector w is perp to both a and v, thus the cross produce of vec_a and vec_v should result in a vector they are BOTH perpendicular to
    vector_w = np.cross(vector_a, vector_v)
    # Rotation around the axis
    if handedness == "left":  # is this the correct assingment or is it left twist that requires this?
        m = -1
    else:
        m = 1
    #########################################################
    # find values for t in terms of parameters
    t = find_rotation(coords, vector_o, vector_a)
    ##############################################################
    # produce helix coordinates
    helix_coords = []
    # for i in len(coords.shape[1]): #for i in t:
    for i in range(0, coords.shape[1]):
        helix_point = starting_point + (vector_a*pitch*t[i]/(np.pi*2)) + radius*(vector_v*np.cos(t[i]) + vector_w*np.sin(t[i]*m))  # m has been removed
        helix_coords.append(list(helix_point))
    helix_coords = np.array(helix_coords)
    return helix_coords

def cost_function_2(estimates, coords, handedness):
    '''input:
        centroid point coordinates
        vector a - direction vector for cylinder 
        vector o - perpendicular vector from origin to axis
        r - radius
       output:
        sum of squared distances (i.e the output of the cost function and what we intend to minimise to fit this cylinder)
    '''
    pitch, radius, ax, ay, az, ox, oy, oz, tx, ty, tz = estimates

    dist = []
    for i in range(0, coords.shape[1]):
        dist_indiv = (coords.T[i] - helix_reconstruction(estimates, handedness, coords)[i])**2
        dist.append(dist_indiv)
    res = np.sum(dist)
    return res

def unpacking_variables(best_fit_values, handedness):
    pitch, radius, ax, ay, az, ox, oy, oz, tx, ty, tz = best_fit_values
    # ensure that the protein is always going from N-C upwards
    if pitch < 0:
        pitch = pitch*-1
        ax = ax*-1
        ay = ay*-1
        az = az*-1
        if handedness == 'left':
            handedness = 'right'
        if handedness == 'right':
            handedness = 'left'
    # repack
    helix_array = np.array([pitch, radius, ax, ay, az, ox, oy, oz, tx, ty, tz])
    cylinder_array = np.array([ax, ay, az, ox, oy, oz, radius])
    # individual values
    vector_a = np.array([ax, ay, az])
    vector_o = np.array([ox, oy, oz])
    terminal_point = np.array([tx, ty, tz])
    return helix_array, cylinder_array, handedness, pitch, radius, vector_a, vector_o, terminal_point


def constraint_1_new(estimates):
    pitch, radius, ax, ay, az, ox, oy, oz, tx, ty, tz = estimates
    a = np.array([ax, ay, az])
    o = np.array([ox, oy, oz])
    return np.dot(a, o)


def constraint_2_new(estimates):
    pitch, radius, ax, ay, az, ox, oy, oz, tx, ty, tz = estimates
    a = np.array([ax, ay, az])
    return np.linalg.norm(a) - 1

def rise_per_repeat(z_coords):
    # Zero coordinates 
    normalised_z_coords = z_coords - z_coords[0]
    # find difference between elements
    z_displacement = [
        j-i for i, j in zip(normalised_z_coords[:-1], normalised_z_coords[1:])]
    mean_dis = np.mean(z_displacement)
    return mean_dis

def total_twist(coords, handedness):
    angles = []
    for i in range(coords.shape[1] - 1):
        point = coords[:, i]
        corresponding = coords[:, i + 1]
        ang1 = np.arctan2(*point[::-1])
        ang2 = np.arctan2(*corresponding[::-1])
        angle = (ang1 - ang2) % (2 * np.pi)
        angles.append(angle)
    if handedness == "left":
        mean_ang = np.mean(angles)
    if handedness == "right":
        new_angles = [abs(angle - 2*np.pi) for angle in angles]
        mean_ang = np.mean(new_angles)
    return mean_ang
