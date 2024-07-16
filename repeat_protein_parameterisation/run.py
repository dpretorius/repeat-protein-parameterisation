import argparse
import os
import numpy as np
from scipy.optimize import minimize
from pdb_processing import read_and_clean_coord, get_ca_coord,find_repeat_centroids, find_Ca_for_second_helix
from helix_fitting import inital_axis_direction, preprocess_data, C, est_rad, find_minimal_perpendicular_vector, constraint_1, constraint_2, cost_function, find_minimal_perpendicular_vector_edit_tobe_general, translate_to_origin, rotate_to_z, rotMat, flatten_coordinates, displacement_along_axis, finding_angles, quadratic_loss_function, helix_reconstruction, constraint_1_new, constraint_2_new, cost_function_2, unpacking_variables, rise_per_repeat, total_twist 
from secondaryhelix_fitting import finding_t_arr, helix_reconstruct_2, frenent_frame, estimate_radius_alpha, helix_3, cost_function_3, estimate_phase
from utils import angle_between
from plotting import plot_helix


def find_arguments():

    parser = argparse.ArgumentParser(description='Parameterising a repeat protein.')

    parser.add_argument("PDB_ID", help="4 digit PDB code for protein", type=str)
    parser.add_argument("Filename", help="Filename for PDB structure, .pdb does not need to be included", type=str)
    parser.add_argument("Chain", help="the chain number of the PDB being investigated", type=str)
    parser.add_argument("First_residue", help="Starting residue of soelnoid region", type=int)
    parser.add_argument("Last_residue", help="Last residue of soelnoid region", type=int)
    parser.add_argument("Repeat_residues", help="Number of residues that make up a solenoid repeat", type=int)
    parser.add_argument("--Ca", nargs='*', help="residue IDs to use for secondary helix", type=int, default=[])
    parser.add_argument("--insert", nargs='*', help="residue IDs to remove", type=int, default=[])
    args = parser.parse_args()
    pdb_id = args.PDB_ID
    filename = args.Filename
    filename_out, file_extension = os.path.splitext(str(os.getcwd()) + "/" + filename)
    if file_extension != ".ent":  # note this has been changed to ent for the RepeatsDB characterisation - make appropriate for .ent or .pdb
        filename = filename_out + ".ent"
    chain_ID = args.Chain
    starting_res = args.First_residue
    ending_res = args.Last_residue
    rep_unit = args.Repeat_residues
    Ca_IDs = args.Ca
    insert_IDs = args.insert
    return pdb_id, chain_ID, filename, filename_out, starting_res, ending_res, rep_unit, Ca_IDs, insert_IDs

def Main():

    # ----------------------------------------
    # User input - execution
    # ----------------------------------------
    pdb_id, chain_ID, filename, filename_out, starting_res, ending_res, rep_unit, Ca_IDs, insert_IDs = find_arguments()

    # ----------------------------------------
    # read, clean and find centroids - execution
    # ----------------------------------------
    # Read in the coordinates from the pdb file
    PDB_coordinate_data, PDB_starting_point = read_and_clean_coord(
        filename, pdb_id, chain_ID, insert_IDs)
    # Get only the Ca coordiantes from the pdb file
    solenoid_Cas = get_ca_coord(
        PDB_coordinate_data, starting_res, ending_res, PDB_starting_point)
    # Find repeat unit centroids
    centroids, centroid_no = find_repeat_centroids(solenoid_Cas, rep_unit)

    # ----------------------------------------
    # main helix inital guesses - execution
    # ----------------------------------------
    # direction vector estimate using Khan method
    w_fit = inital_axis_direction(centroids)

    data_attempt, X = preprocess_data(centroids)

    # Find center
    C_fit = C(w_fit, data_attempt) + X

    # Find radius
    r_fit = est_rad(w_fit, data_attempt)

    # ----------------------------------------
    # minimise main helix parameters - execution
    # ----------------------------------------
    # Find perpendicular vector from origin
    vec_o = find_minimal_perpendicular_vector(w_fit, C_fit)

    # Prepare inital guess of a_, o_ and r
    estimate = [w_fit[0], w_fit[1], w_fit[2],
                vec_o[0], vec_o[1], vec_o[2], r_fit]

    # contraints
    cons = ({'type': 'eq', 'fun': constraint_1},
            {'type': 'eq', 'fun': constraint_2})

    bestFitValues = minimize(cost_function, estimate, args=(
        centroids), constraints=cons, method='trust-constr')
    inital_best_fit = np.array(bestFitValues.x)
    ax, ay, az, ox, oy, oz, r = inital_best_fit

    w_fit_updated = np.array([ax, ay, az])
    vec_o_updated = np.array([ox, oy, oz])
    radius_estimate = r

    # ----------------------------------------
    # translation and rotation - execution
    # ----------------------------------------
    # make a centerline
    linepts = (w_fit_updated) * np.mgrid[-5:5:2j][:, np.newaxis]
    linepts += vec_o_updated

    # find the projected point on the central axis
    projected_first_point = find_minimal_perpendicular_vector_edit_tobe_general(
        w_fit_updated, vec_o_updated, centroids)

    # translation to origin
    translated_axis, translated_centroids, translated_solenoid_Cas = translate_to_origin(
        linepts, centroids, solenoid_Cas, projected_first_point)

    # rotation
    # Get eular angles by rotating the direction vector to the z-axis
    alpha, beta = rotate_to_z(w_fit_updated, translated_centroids[0][2])

    # Get the rotation matrix
    rot_matrix = (rotMat('y',  beta)) @ rotMat('x', alpha)

    # Rotate 1) central axis, 2) Cas coords and 3) direction vector to align with the z axis
    new_centroids_old = rot_matrix @ (translated_centroids.T)

    # ----------------------------------------
    # pitch and handedness estimation - execution
    # ----------------------------------------
    # Preprocess input data for least squares linear regression
    xy_centroid, z_centroid = flatten_coordinates(new_centroids_old)

    # find z displacement
    z_displacement = displacement_along_axis(z_centroid)

    # find rotation around central axis AND determine handedness
    rotation_around_axis, handedness = finding_angles(xy_centroid)

    # Quadratic loss function
    gradient = quadratic_loss_function(rotation_around_axis, z_displacement)

    # because when rotation=2pi then displacement=pitch
    pitch_estimate = gradient * 2*np.pi

    # ----------------------------------------
    # full minimisation - execution
    # ----------------------------------------
    centroids = centroids.T

    # prepare estimates of P, r, a, o, and t0
    first_data_point = centroids[:, 0]
    estimates = [pitch_estimate, radius_estimate,
                 w_fit_updated[0], w_fit_updated[1], w_fit_updated[2],
                 vec_o_updated[0], vec_o_updated[1], vec_o_updated[2],
                 first_data_point[0], first_data_point[1], first_data_point[2]]  # pretty sure the first data point is able to be reduced to just 1 not 3 |(rotational value)

    helix_coordinates = helix_reconstruction(estimates, handedness, centroids)

    cons_2 = ({'type': 'eq', 'fun': constraint_1_new},
              {'type': 'eq', 'fun': constraint_2_new})

    bestFitValues = minimize(cost_function_2, estimates, args=(
        centroids, handedness), constraints=cons_2)
    best_fit = np.array(bestFitValues.x)
    best_fit_values, cylinder_array, handedness, pitch, radius, vector_a, vector_o, terminal_point = unpacking_variables(
        best_fit, handedness)

    print("")
    print("Superhelix parameters")
    print("Pitch = {:.5f} Å".format(pitch))
    print("Radius = {:.5f} Å".format(radius))
    print("Handedness = ", handedness)
    print("Direction vector = ", vector_a)

    new_projected_first_point = find_minimal_perpendicular_vector_edit_tobe_general(
        vector_a, vector_o, centroids.T)

    new_helix = helix_reconstruction(best_fit_values, handedness, centroids)

    #RMSD calculations
    helix_residual = cost_function_2(best_fit_values, centroids, handedness)
    helix_rmsd = np.sqrt((helix_residual)/centroids.shape[1])
    print("helix RMSD:", helix_rmsd)
    cylinder_residual = cost_function(cylinder_array, centroids.T)
    cylinder_rmsd = np.sqrt((cylinder_residual)/centroids.shape[1])
    print("cylinder RMSD:", cylinder_rmsd)

    # translate 
    translated_axis, translated_centroids, translated_solenoid_Cas = translate_to_origin(
        linepts, centroids.T, solenoid_Cas, new_projected_first_point)

    helix_coordinates = helix_coordinates - new_projected_first_point

    new_helix = new_helix - new_projected_first_point

    # rotation
    # Get eular angles by rotating the direction vector to the z-axis
    alpha, beta = rotate_to_z(vector_a, translated_centroids[0][2])

    # Get the rotation matrix
    rot_matrix = (rotMat('y',  beta)) @ rotMat('x', alpha)

    # Rotate
    new_centroids = rot_matrix @ (translated_centroids.T)
    new_Ca_coords = rot_matrix @ (translated_solenoid_Cas.T)
    new_new_helix = rot_matrix @ (new_helix.T)

    # prepare helix for further analysis
    xy_helix, z_helix = flatten_coordinates(new_new_helix)

    # rise per repeat
    rise_per_rep = rise_per_repeat(z_helix)
    print("Rise = {:.5f} Å".format(rise_per_rep))

    # twist
    twist = total_twist(xy_helix, handedness)
    print("Twist = {:.5f} Å".format(twist))
    print("")

    # ----------------------------------------
    # superhelix - functions
    # ----------------------------------------
    # get the Ca residues that will be used as the basis for the local helix
    Ca_residues = find_Ca_for_second_helix(
        solenoid_Cas.T, rep_unit, Ca_IDs, insert_IDs)  # not new_Ca_coords

    # find the t values via  (projection of Ca residues onto the major helix)
    t_arr = finding_t_arr(best_fit_values, handedness,
                          Ca_residues.T, centroids)

    # producing inital helix
    test_helix = helix_reconstruct_2(t_arr, best_fit_values, handedness)

    # find the frenent frame for each point on the helix
    T, B, N = frenent_frame(test_helix)

    alpha_est = estimate_radius_alpha(Ca_residues, test_helix)
    omega_est = 0
    phase_est = estimate_phase(test_helix, Ca_residues)

    local_est = omega_est, alpha_est, phase_est

    # plotting of the secondary helix with inital estimates
    double_helix = helix_3(local_est, Ca_residues.T, test_helix, t_arr, B, N)

    # minimisation of the secondary helix to the Ca points
    bounds = (((-2*np.pi), (2*np.pi)), (0, None), (0, (2*np.pi)))
    bestFitValues = minimize(cost_function_3, local_est, args=(
        Ca_residues, test_helix, t_arr, B, N), bounds=bounds)
    refined_est = bestFitValues.x
    omega, alpha, phase = refined_est

    if omega < 0:
        local_handedness = "left"
    if omega > 0:
        local_handedness = "right"

    print("")
    print("Local helix parameters")
    print("Omega = {:.2f} what unit is this?".format(omega))
    print("Alpha = {:.2f} Å".format(alpha))
    print("Phase = {:.2f} radians".format(phase))
    print("Hand = ", local_handedness)

    local_helix_residual = cost_function_3(
        refined_est, Ca_residues, test_helix, t_arr, B, N)
    local_helix_rmsd = np.sqrt((local_helix_residual)/Ca_residues.T.shape[1])
    print("local helix RMSD:", local_helix_rmsd)
    print("")

    final_helix = helix_3(refined_est, Ca_residues.T, test_helix, t_arr, B, N)

    final_helix = rot_matrix @ ((final_helix - new_projected_first_point).T)
    double_helix = rot_matrix @ ((double_helix - new_projected_first_point).T)
    test_helix = rot_matrix @ ((test_helix.T - new_projected_first_point).T)
    Ca_residues = rot_matrix @ ((Ca_residues - new_projected_first_point).T)

    # ----------------------------------------
    # Plot parameterised protein
    # ----------------------------------------
    
    plot_helix(new_centroids, new_new_helix, Ca_residues, final_helix, new_Ca_coords)


if __name__ == "__main__":
    Main()
