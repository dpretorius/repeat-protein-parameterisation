import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d
import matplotlib

def plot_helix(new_centroids, new_helix, Ca_residues, final_helix, new_Ca_coords):
#    matplotlib.rcParams.update({'font.size': 10})
#    matplotlib.rcParams['font.family'] = "Times New Roman"
#    ax = m3d.Axes3D(plt.figure())

    # plot main helix
#    ax.scatter3D(*new_centroids, color='firebrick', label="Centroids", s=30)
#    ax.scatter(*new_helix, marker='x', color='firebrick', label='Main helix')
#    ax.plot3D(*new_helix, color='firebrick', linewidth=1)

    # plot superhelix
#    ax.scatter3D(*Ca_residues, color='royalblue', label=r"Starting C$\alpha$", s=30)
#    ax.scatter(*final_helix, marker='x', color='royalblue', label='Superhelix')
#    ax.plot3D(*final_helix, color='royalblue', linewidth=1)

    # Plot Cα backbone
#    ax.plot3D(*new_Ca_coords, color='black', linestyle='dashed', linewidth=0.5, label="Backbone")
#    ax.set_xlabel('x')
#    ax.set_ylabel('y')
#    ax.set_zlabel('z')
#    ax.legend()
#    ax.view_init(elev=10., azim=257)
#    plt.show()
    
    
    # Plot parameterised protein
    matplotlib.rcParams.update({'font.size': 10})
    matplotlib.rcParams['font.family'] = "Times New Roman"
    ax = m3d.Axes3D(plt.figure())

    # plot main helix
    ax.scatter3D(*new_centroids, color='firebrick', label="Centroids", s=30)
    ax.scatter(*new_helix, marker='x', color='firebrick', label='Main helix')
    ax.plot3D(*new_helix, color='firebrick', linewidth=1)

    # plot superhelix
    ax.scatter3D(*Ca_residues, color='royalblue', label=r"Starting C$\alpha$", s=30)
    ax.scatter(*final_helix, marker='x', color='royalblue', label='Superhelix')
    ax.plot3D(*final_helix, color='royalblue', linewidth=1)

    #Plot C⍺ backbone
    ax.plot3D(*new_Ca_coords, color='black',  linestyle='dashed',linewidth=0.5, label="Backbone")

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.legend()
    ax.view_init(elev=10., azim=257)

    plt.show()