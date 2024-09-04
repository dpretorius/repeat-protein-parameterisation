import argparse
import os
import numpy as np
from csv import writer
from scipy.optimize import minimize
from pdb_processing import read_and_clean_coord, get_ca_coord, find_repeat_centroids, find_Ca_for_second_helix
from helix_fitting import (
    inital_axis_direction, preprocess_data, C, est_rad,
    find_minimal_perpendicular_vector, constraint_1, constraint_2, cost_function,
    find_minimal_perpendicular_vector_edit_tobe_general, translate_to_origin, rotate_to_z,
    rotMat, flatten_coordinates, displacement_along_axis, finding_angles,
    quadratic_loss_function, helix_reconstruction, constraint_1_new,
    constraint_2_new, cost_function_2, unpacking_variables, rise_per_repeat, total_twist
)
from secondaryhelix_fitting import (
    finding_t_arr, helix_reconstruct_2, frenent_frame, estimate_radius_alpha,
    helix_3, cost_function_3, estimate_phase
)
from plotting import plot_helix

from uncertainty import (
    gradient_function, calculate_uncertainties_from_grad, numerical_jacobian,
    propagate_uncertainty, gradient_function_3
)

from utils import append_list_as_row

def generate_perfect_helix(num_points, pitch, radius):
    t = np.linspace(0, 4 * np.pi, num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = (pitch / (2 * np.pi)) * t
    return np.vstack((x, y, z)).T

def generate_noisy_helix(num_points, pitch, radius, noise_level=0):
    t = np.linspace(0, 4 * np.pi, num_points)
    x = radius * np.cos(t) + np.random.normal(0, noise_level, num_points)
    y = radius * np.sin(t) + np.random.normal(0, noise_level, num_points)
    z = (pitch / (2 * np.pi)) * t + np.random.normal(0, noise_level, num_points)
    return np.vstack((x, y, z)).T

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
    if file_extension != ".ent":  # note this has been changed to ent for the RepeatsDB characterisation
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
    
    # Fake perfect data
    #centroid_no = 10
    #pitch = 10
    #radius = 5
    #centroids = generate_noisy_helix(centroid_no, pitch, radius)

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
        centroids), constraints=cons) #method='trust-constr'
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
                 first_data_point[0], first_data_point[1], first_data_point[2]]

    helix_coordinates = helix_reconstruction(estimates, handedness, centroids)

    cons_2 = ({'type': 'eq', 'fun': constraint_1_new},
              {'type': 'eq', 'fun': constraint_2_new})
    
    bestFitValues = minimize(cost_function_2, estimates, args=(
        centroids, handedness), constraints=cons_2, method="SLSQP")
    best_fit = np.array(bestFitValues.x)
    best_fit_values, cylinder_array, handedness, pitch, radius, vector_a, vector_o, terminal_point = unpacking_variables(
        best_fit, handedness)
    
    best_fit = bestFitValues.x
    fun = bestFitValues.fun
    grad = lambda params: gradient_function(params, centroids, handedness)
    uncertainties, cov = calculate_uncertainties_from_grad(grad, best_fit, fun, len(best_fit))

    print("")
    print("Superhelix parameters")
    print(f"Pitch = {pitch:.5f} Å ± {uncertainties[0]:.5f}")
    print(f"Radius = {radius:.5f} Å ± {uncertainties[1]:.5f}")
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

    # twist
    twist = np.degrees(total_twist(xy_helix, handedness))
    
    # rise per repeat
    rise_jac = numerical_jacobian(lambda x: rise_per_repeat(flatten_coordinates(helix_reconstruction(x, handedness, centroids))[1]), best_fit)
    rise_uncertainty = propagate_uncertainty(cov, rise_jac)
    print(f"Rise = {rise_per_rep:.5f} Å ± {rise_uncertainty:.5f}")

    # twist
    twist_jac = numerical_jacobian(lambda x: total_twist(flatten_coordinates(helix_reconstruction(x, handedness, centroids))[0], handedness), best_fit)
    twist_uncertainty = propagate_uncertainty(cov, twist_jac)
    print(f"Twist = {twist:.5f} degrees ± {twist_uncertainty:.5f} degrees")
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
    
    # Calculate uncertainties
    grad = lambda params: gradient_function_3(params, Ca_residues, test_helix, t_arr, B, N)
    uncertainties, cov = calculate_uncertainties_from_grad(grad, refined_est, bestFitValues.fun, len(refined_est))

    omega_deg = omega
    phase_deg = np.degrees(phase)

    print("")
    print("Local helix parameters")
    print(f"Omega = {omega_deg:.2f} radians ± {uncertainties[0]:.2f} radians")
    print(f"Alpha = {alpha:.2f} Å ± {uncertainties[1]:.2f} Å")
    print(f"Phase = {phase_deg:.2f} degrees ± {np.degrees(uncertainties[2]):.2f} degrees")
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
    
    # Save data to CSV
    header = ["Type", "Description", "PDB ID", "Chain", "First Residue", "Last Residue", "Repeat Residues", 
              "Centroid Number", "Pitch (Å)", "Pitch Uncertainty (Å)", "Radius (Å)", "Radius Uncertainty (Å)", 
              "Rise (Å)", "Rise Uncertainty (Å)", "Twist (degrees)", "Twist Uncertainty (degrees)", "Handedness", 
              "Cylinder RMSD", "Helix RMSD", "Omega (radians)", "Omega Uncertainty (radians)", "Alpha (Å)", 
              "Alpha Uncertainty (Å)", "Superhelix Handedness", "Superhelix RMSD"]

    row_contents = ["Elongated", "beta solenoid", str(pdb_id), str(chain_ID), str(starting_res), str(ending_res), 
                    str(rep_unit), str(centroid_no), f'{pitch:.3f}', f'{uncertainties[0]:.3f}', f'{radius:.3f}', 
                    f'{uncertainties[1]:.3f}', f'{rise_per_rep:.3f}', f'{rise_uncertainty:.3f}', f'{twist:.3f}', 
                    f'{twist_uncertainty:.3f}', str(handedness), f'{cylinder_rmsd:.3f}', f'{helix_rmsd:.3f}', 
                    f'{omega_deg:.3f}', f'{uncertainties[0]:.3f}', f'{alpha:.3f}', f'{uncertainties[1]:.3f}', 
                    str(local_handedness), f'{local_helix_rmsd:.3f}']
    
    file_name = 'solenoid_parameters_and_uncertainty_all.csv'
    
    # Check if file exists and write header if it doesn't
    if not os.path.exists(file_name):
        with open(file_name, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(header)
    
    # Append the data row
    append_list_as_row(file_name, row_contents)


if __name__ == "__main__":
    Main()
