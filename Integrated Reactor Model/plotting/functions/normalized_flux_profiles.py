"""Functions for plotting normalized flux profiles."""

import os
import numpy as np
import matplotlib.pyplot as plt
from inputs import inputs
from eigenvalue.tallies.normalization import calc_norm_factor
import openmc

def create_flux_profile_plots(flux_mean, shape, plot_dir, filename, title_prefix="",
                            active_core_radius=None, tank_radius=None, reflector_radius=None, half_height=None,
                            group_fluxes=None):
    """Helper function to create the standard 3-panel flux profile plot.

    Parameters
    ----------
    flux_mean : numpy.ndarray
        Mean flux values in shape [nx, ny, nz]
    shape : list
        Shape of the mesh [nx, ny, nz]
    plot_dir : str
        Directory to save the plot
    filename : str
        Name of the output file
    title_prefix : str, optional
        Prefix to add to plot titles
    active_core_radius : float
        Radius of active core region in cm
    tank_radius : float
        Radius of tank in cm
    reflector_radius : float
        Radius of reflector in cm
    half_height : float
        Half height of the core in cm
    group_fluxes : dict, optional
        Dictionary containing energy group fluxes with keys 'thermal', 'epithermal', 'fast'
    """
    # Create figure with subplots
    n_rows = 2 if group_fluxes else 1
    fig = plt.figure(figsize=(24, 8*n_rows))
    gs = plt.GridSpec(n_rows, 3, figure=fig)

    # First row - total flux profiles
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Create twin axes for absolute and normalized flux for both radial plots
    ax0_norm = ax0.twinx()
    ax1_norm = ax1.twinx()

    # Plot radial profiles (both midplane and axially averaged)
    center_x = shape[1] // 2  # X is second index
    center_y = shape[2] // 2  # Y is third index
    center_z = shape[0] // 2  # Z is first index

    # Calculate axially averaged profiles
    # X-direction axially averaged profile (at Y=0)
    avg_flux_x = np.mean(flux_mean[:, :, center_y], axis=0)  # Average along Z, vary along X
    x = np.linspace(-reflector_radius, reflector_radius, len(avg_flux_x))

    # Y-direction axially averaged profile (at X=0)
    avg_flux_y = np.mean(flux_mean[:, center_x, :], axis=0)  # Average along Z, vary along Y
    y = np.linspace(-reflector_radius, reflector_radius, len(avg_flux_y))

    # 45-degree diagonal profiles for axially averaged
    max_offset = min(center_x, center_y, shape[1]-center_x-1, shape[2]-center_y-1)
    diagonal_coords = np.arange(-max_offset, max_offset+1)
    r_diag = diagonal_coords * np.sqrt(2) * (reflector_radius / (shape[1]//2))

    # Get axially averaged diagonal fluxes
    avg_flux_diag1 = np.mean(flux_mean[:,
                                     center_x + diagonal_coords,
                                     center_y + diagonal_coords], axis=0)
    avg_flux_diag2 = np.mean(flux_mean[:,
                                     center_x + diagonal_coords,
                                     center_y - diagonal_coords], axis=0)

    # Find peak flux across all axially averaged radial profiles
    peak_avg_radial_flux = np.max([np.max(avg_flux_x),
                                  np.max(avg_flux_y),
                                  np.max(avg_flux_diag1),
                                  np.max(avg_flux_diag2)])

    # Plot axially averaged absolute fluxes
    ax0.plot(x, avg_flux_x, 'b-', label='X Direction (Y=0)')
    ax0.plot(y, avg_flux_y, 'r-', label='Y Direction (X=0)')
    ax0.plot(r_diag, avg_flux_diag1, 'g-', label='45° Direction')
    ax0.plot(r_diag, avg_flux_diag2, 'm-', label='135° Direction')

    # Plot axially averaged normalized fluxes
    ax0_norm.plot(x, avg_flux_x/peak_avg_radial_flux, 'b-')
    ax0_norm.plot(y, avg_flux_y/peak_avg_radial_flux, 'r-')
    ax0_norm.plot(r_diag, avg_flux_diag1/peak_avg_radial_flux, 'g-')
    ax0_norm.plot(r_diag, avg_flux_diag2/peak_avg_radial_flux, 'm-')

    # Add boundaries to axially averaged plot
    for ax in [ax0, ax0_norm]:
        ax.axvline(x=active_core_radius, color='red', linestyle='--', label='Active Core')
        ax.axvline(x=-active_core_radius, color='red', linestyle='--')
        ax.axvline(x=tank_radius, color='blue', linestyle='--', label='Tank')
        ax.axvline(x=-tank_radius, color='blue', linestyle='--')
        ax.axvline(x=reflector_radius, color='green', linestyle='--', label='Reflector')
        ax.axvline(x=-reflector_radius, color='green', linestyle='--')

    # Configure axially averaged axes
    ax0.set_title(f'{title_prefix}Axially Averaged Radial Flux Profiles')
    ax0.set_xlabel('Distance from Center [cm]')
    ax0.set_ylabel('Absolute Flux [n/cm²/s]')
    ax0_norm.set_ylabel('Normalized Flux')
    ax0.grid(True)

    # Get midplane profiles
    radial_flux_x = flux_mean[center_z, :, center_y]
    radial_flux_y = flux_mean[center_z, center_x, :]

    # Calculate midplane diagonal profiles with bounds checking
    radial_flux_diag1 = flux_mean[center_z,
                                 center_x + diagonal_coords,
                                 center_y + diagonal_coords]
    radial_flux_diag2 = flux_mean[center_z,
                                 center_x + diagonal_coords,
                                 center_y - diagonal_coords]

    peak_radial_flux = np.max([np.max(radial_flux_x),
                              np.max(radial_flux_y),
                              np.max(radial_flux_diag1),
                              np.max(radial_flux_diag2)])

    # Plot midplane profiles
    ax1.plot(x, radial_flux_x, 'b-', label='X Direction (Y=0)')
    ax1.plot(y, radial_flux_y, 'r-', label='Y Direction (X=0)')
    ax1.plot(r_diag, radial_flux_diag1, 'g-', label='45° Direction')
    ax1.plot(r_diag, radial_flux_diag2, 'm-', label='135° Direction')

    ax1_norm.plot(x, radial_flux_x/peak_radial_flux, 'b-')
    ax1_norm.plot(y, radial_flux_y/peak_radial_flux, 'r-')
    ax1_norm.plot(r_diag, radial_flux_diag1/peak_radial_flux, 'g-')
    ax1_norm.plot(r_diag, radial_flux_diag2/peak_radial_flux, 'm-')

    # Add boundaries to midplane plot
    for ax in [ax1, ax1_norm]:
        ax.axvline(x=active_core_radius, color='red', linestyle='--', label='Active Core')
        ax.axvline(x=-active_core_radius, color='red', linestyle='--')
        ax.axvline(x=tank_radius, color='blue', linestyle='--', label='Tank')
        ax.axvline(x=-tank_radius, color='blue', linestyle='--')
        ax.axvline(x=reflector_radius, color='green', linestyle='--', label='Reflector')
        ax.axvline(x=-reflector_radius, color='green', linestyle='--')

    # Configure midplane axes
    ax1.set_title(f'{title_prefix}Radial Flux Profiles (Z Mid-plane)')
    ax1.set_xlabel('Distance from Center [cm]')
    ax1.set_ylabel('Absolute Flux [n/cm²/s]')
    ax1_norm.set_ylabel('Normalized Flux')
    ax1.grid(True)

    # Calculate positions in mesh indices with bounds checking
    core_center_x = shape[1] // 2
    core_center_y = shape[2] // 2

    # Calculate indices ensuring they don't exceed mesh boundaries
    max_idx = shape[1] // 2  # Maximum valid offset from center

    # Calculate assembly edge index and ensure it's within bounds
    assembly_edge_idx = min(
        int((active_core_radius) / reflector_radius * shape[1] // 2),
        max_idx - 1
    )

    # Calculate other indices relative to assembly edge
    half_fuel_idx = assembly_edge_idx // 2

    # Ensure core edge index is within bounds
    core_edge_idx = min(
        int(tank_radius / reflector_radius * shape[1] // 2),
        max_idx - 1
    )

    # Calculate position halfway between fuel edge and core edge
    fuel_to_core_edge_idx = assembly_edge_idx + min(
        (core_edge_idx - assembly_edge_idx) // 2,
        max_idx - assembly_edge_idx - 1
    )

    # Get axial profiles at different radial positions
    z = np.linspace(-half_height, half_height, shape[0])

    # Function to get average flux around a circular path at given radius
    def get_circular_average_flux(flux_array, radius_idx):
        if radius_idx == 0:
            return flux_array[:, core_center_x, core_center_y]

        # Create a circle of points at the given radius
        theta = np.linspace(0, 2*np.pi, 360)  # 1 degree resolution
        x_indices = np.round(core_center_x + radius_idx * np.cos(theta)).astype(int)
        y_indices = np.round(core_center_y + radius_idx * np.sin(theta)).astype(int)

        # Filter out points that would be out of bounds
        valid_points = (x_indices >= 0) & (x_indices < shape[1]) & \
                      (y_indices >= 0) & (y_indices < shape[2])
        x_indices = x_indices[valid_points]
        y_indices = y_indices[valid_points]

        # Get flux at all valid points on the circle for each z
        circular_fluxes = np.zeros((shape[0], len(x_indices)))
        for i, (x_idx, y_idx) in enumerate(zip(x_indices, y_indices)):
            circular_fluxes[:, i] = flux_array[:, x_idx, y_idx]

        # Average across all points on the circle
        return np.mean(circular_fluxes, axis=1)

    # Calculate averaged axial profiles at different radial positions
    axial_flux_center = get_circular_average_flux(flux_mean, 0)
    axial_flux_half = get_circular_average_flux(flux_mean, half_fuel_idx)
    axial_flux_fuel_edge = get_circular_average_flux(flux_mean, assembly_edge_idx)
    axial_flux_mid_to_core = get_circular_average_flux(flux_mean, fuel_to_core_edge_idx)
    axial_flux_core_edge = get_circular_average_flux(flux_mean, core_edge_idx)

    # Find the peak flux across all axial profiles for normalization
    peak_flux = np.max([np.max(axial_flux_center),
                       np.max(axial_flux_half),
                       np.max(axial_flux_fuel_edge),
                       np.max(axial_flux_mid_to_core),
                       np.max(axial_flux_core_edge)])

    # Plot each profile normalized to the overall peak flux
    ax2.plot(axial_flux_center / peak_flux, z, 'b-',
             label='Core Center (r=0)')

    ax2.plot(axial_flux_half / peak_flux, z, 'g-',
             label=f'50% to Core Edge (r={active_core_radius/2:.1f} cm)')

    ax2.plot(axial_flux_fuel_edge / peak_flux, z, 'r-',
             label=f'Core Edge (r={active_core_radius:.1f} cm)')

    ax2.plot(axial_flux_mid_to_core / peak_flux, z, 'm-',
             label=f'50% Core Edge to Tank Edge (r={(active_core_radius + tank_radius)/2:.1f} cm)')

    ax2.plot(axial_flux_core_edge / peak_flux, z, 'k-',
             label=f'Tank Edge (r={tank_radius:.1f} cm)')

    # Add top axis for absolute flux values
    ax2_abs = ax2.twiny()
    ax2_abs.plot(axial_flux_center, z, 'b-')
    ax2_abs.plot(axial_flux_half, z, 'g-')
    ax2_abs.plot(axial_flux_fuel_edge, z, 'r-')
    ax2_abs.plot(axial_flux_mid_to_core, z, 'm-')
    ax2_abs.plot(axial_flux_core_edge, z, 'k-')
    ax2_abs.set_xlabel('Absolute Flux [n/cm²/s]')

    # Add fuel height lines
    ax2.axhline(y=half_height, color='gray', linestyle='--', label='Fuel Boundary')
    ax2.axhline(y=-half_height, color='gray', linestyle='--')

    ax2.set_title(f'{title_prefix}Normalized Axial Flux Profiles')
    ax2.set_xlabel('Normalized Flux')
    ax2.set_ylabel('Height [cm]')
    ax2.grid(True)

    # Add legend for axial profiles below the third subplot
    ax2.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)

    # Add legends below the first two plots
    # Create a common legend for both radial plots
    lines0, labels0 = ax0.get_legend_handles_labels()
    fig.legend(lines0[:4], labels0[:4],
              bbox_to_anchor=(0.35, 0.54), loc='lower center', ncol=4)
    # Add boundary legend
    fig.legend([plt.Line2D([0], [0], color=c, linestyle='--') for c in ['red', 'blue', 'green']],
              ['Active Core', 'Tank', 'Reflector'],
              bbox_to_anchor=(0.35, 0.52), loc='lower center', ncol=3)

    if group_fluxes:
        # Second row - energy group specific profiles
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[1, 2])

        # Create twin axes for the second row
        ax3_norm = ax3.twinx()
        ax4_norm = ax4.twinx()

        colors = {'thermal': 'b', 'epithermal': 'g', 'fast': 'r'}
        location_colors = {'center': 'b', 'half': 'g', 'edge': 'r', 'mid_to_core': 'm', 'core_edge': 'k'}
        location_names = {
            'center': 'Core Center (r=0)',
            'half': f'50% to Core Edge (r={active_core_radius/2:.1f} cm)',
            'edge': f'Core Edge (r={active_core_radius:.1f} cm)',
            'mid_to_core': f'50% Core Edge to Tank Edge (r={(active_core_radius + tank_radius)/2:.1f} cm)',
            'core_edge': f'Tank Edge (r={tank_radius:.1f} cm)'
        }
        energy_linestyles = {'thermal': '-', 'epithermal': '--', 'fast': ':'}

        # Plot energy group specific axially averaged profiles
        for group_name, group_flux in group_fluxes.items():
            # Calculate axially averaged profiles for this group
            avg_flux_x = np.mean(group_flux[:, :, center_y], axis=0)
            avg_flux_y = np.mean(group_flux[:, center_x, :], axis=0)
            avg_flux_diag1 = np.mean(group_flux[:,
                                              center_x + diagonal_coords,
                                              center_y + diagonal_coords], axis=0)
            avg_flux_diag2 = np.mean(group_flux[:,
                                              center_x + diagonal_coords,
                                              center_y - diagonal_coords], axis=0)

            # Plot on ax3
            ax3.plot(x, avg_flux_x, color=colors[group_name], linestyle='-',
                    label=f'{group_name.capitalize()} (All Directions)')
            ax3.plot(y, avg_flux_y, color=colors[group_name], linestyle='-')
            ax3.plot(r_diag, avg_flux_diag1, color=colors[group_name], linestyle='-')
            ax3.plot(r_diag, avg_flux_diag2, color=colors[group_name], linestyle='-')

            # Plot midplane profiles for this group
            radial_flux_x = group_flux[center_z, :, center_y]
            radial_flux_y = group_flux[center_z, center_x, :]
            radial_flux_diag1 = group_flux[center_z,
                                         center_x + diagonal_coords,
                                         center_y + diagonal_coords]
            radial_flux_diag2 = group_flux[center_z,
                                         center_x + diagonal_coords,
                                         center_y - diagonal_coords]

            # Plot on ax4
            ax4.plot(x, radial_flux_x, color=colors[group_name], linestyle='-',
                    label=f'{group_name.capitalize()} (All Directions)')
            ax4.plot(y, radial_flux_y, color=colors[group_name], linestyle='-')
            ax4.plot(r_diag, radial_flux_diag1, color=colors[group_name], linestyle='-')
            ax4.plot(r_diag, radial_flux_diag2, color=colors[group_name], linestyle='-')

        # Plot axial profiles with all locations and energy groups
        for loc_name, loc_color in location_colors.items():
            # Get the appropriate radius index
            if loc_name == 'center':
                radius_idx = 0
            elif loc_name == 'half':
                radius_idx = half_fuel_idx
            elif loc_name == 'edge':
                radius_idx = assembly_edge_idx
            elif loc_name == 'mid_to_core':
                radius_idx = fuel_to_core_edge_idx
            else:  # core_edge
                radius_idx = core_edge_idx

            # Plot each energy group for this location
            for group_name, group_flux in group_fluxes.items():
                axial_flux = get_circular_average_flux(group_flux, radius_idx)
                ax5.plot(axial_flux, z, color=loc_color, linestyle=energy_linestyles[group_name],
                        label=location_names[loc_name] if group_name == 'thermal' else '')

        # Add boundaries to energy group plots
        for ax in [ax3, ax3_norm, ax4, ax4_norm]:
            ax.axvline(x=active_core_radius, color='red', linestyle='--', label='Active Core')
            ax.axvline(x=-active_core_radius, color='red', linestyle='--')
            ax.axvline(x=tank_radius, color='blue', linestyle='--', label='Tank')
            ax.axvline(x=-tank_radius, color='blue', linestyle='--')
            ax.axvline(x=reflector_radius, color='green', linestyle='--', label='Reflector')
            ax.axvline(x=-reflector_radius, color='green', linestyle='--')

        # Add fuel height lines to axial plot
        ax5.axhline(y=half_height, color='gray', linestyle='--', label='Fuel Boundary')
        ax5.axhline(y=-half_height, color='gray', linestyle='--')

        # Configure energy group plot axes
        ax3.set_title('Energy Group Axially Averaged Radial Flux Profiles')
        ax3.set_xlabel('Distance from Center [cm]')
        ax3.set_ylabel('Flux [n/cm²/s]')
        ax3.grid(True)
        ax3.legend(bbox_to_anchor=(0.5, -0.09), loc='upper center', ncol=3)

        ax4.set_title('Energy Group Radial Flux Profiles (Z Mid-plane)')
        ax4.set_xlabel('Distance from Center [cm]')
        ax4.set_ylabel('Flux [n/cm²/s]')
        ax4.grid(True)
        ax4.legend(bbox_to_anchor=(0.5, -0.09), loc='upper center', ncol=3)

        ax5.set_title('Energy Group Axial Flux Profiles')
        ax5.set_xlabel('Flux [n/cm²/s]')
        ax5.set_ylabel('Height [cm]')
        ax5.grid(True)
        # Add legend for locations
        ax5.legend(bbox_to_anchor=(0.5, -0.09), loc='upper center', ncol=2)
        # Add line style legend
        fig.text(0.85, 0.3, 'Line styles:\nSolid = Thermal\nDashed = Epithermal\nDotted = Fast',
                bbox=dict(facecolor='white', alpha=0.8))

    # Adjust subplot spacing
    plt.subplots_adjust(bottom=0.2, wspace=0.4, hspace=0.4 if group_fluxes else 0.2)

    plt.savefig(os.path.join(plot_dir, filename),
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_normalized_flux_profiles(sp, plot_dir):
    """Plot normalized radial and axial flux profiles.

    Parameters
    ----------
    sp : openmc.StatePoint
        StatePoint file containing the tally results
    plot_dir : str
        Directory to save the plots
    """
    # Get the mesh tally
    mesh_tally = sp.get_tally(name='flux_mesh')
    flux = mesh_tally.get_slice(scores=['flux'])

    # Calculate actual core dimensions based on maximum fuel assembly row
    core_layout = np.array(inputs['core_lattice'])
    # Count actual fuel assemblies (F and E positions) in each row
    max_fuel_assemblies = max(sum(1 for pos in row if pos in ['F', 'E']) for row in core_layout)

    # Calculate the side length based on assembly type
    if inputs['assembly_type'] == 'Plate':
        assembly_unit_width = (inputs['fuel_plate_width'] + 2*inputs['clad_structure_width']) * 100  # cm
    else:  # Pin type
        assembly_unit_width = (inputs['pin_pitch'] * inputs['n_side_pins']) * 100  # cm

    total_width = max_fuel_assemblies * assembly_unit_width
    half_width = total_width / 2

    # Get dimensions from inputs (in cm)
    active_core_radius = half_width  # Use calculated fuel region radius for core lines
    tank_radius = inputs['tank_radius'] * 100  # Tank outer boundary
    reflector_radius = tank_radius + inputs['reflector_thickness'] * 100  # Reflector outer boundary
    half_height = inputs['fuel_height'] * 50  # Just use fuel height

    # Get mesh dimensions from the tally
    mesh_filter = mesh_tally.find_filter(openmc.MeshFilter)
    shape = mesh_filter.mesh.dimension

    # Calculate mesh volume
    mesh_volume = (2 * reflector_radius / shape[0]) * (2 * reflector_radius / shape[1]) * (2 * half_height / shape[2])

    # Create numpy arrays from tally data and normalize
    power_mw = inputs.get('core_power', 1.0)
    norm_factor = calc_norm_factor(power_mw, sp)

    if inputs['Core_Three_Group_Energy_Bins']:
        # Create directory for energy group plots
        energy_group_dir = os.path.join(plot_dir, '3_energy_group_maps')
        os.makedirs(energy_group_dir, exist_ok=True)

        # Get energy boundaries
        energy_bounds = [0.0, inputs['thermal_cutoff'], inputs['fast_cutoff'], 20.0e6]
        group_names = ['thermal', 'epithermal', 'fast']

        # Process each energy group
        n_spatial = np.prod(shape)
        mean_data = flux.mean.reshape(n_spatial, -1)

        # Create dictionary to store group fluxes
        group_fluxes = {}

        # Process each energy group
        for group_idx, group_name in enumerate(group_names):
            # Extract and reshape data for this group
            group_mean = mean_data[:, group_idx].reshape(shape) * norm_factor / mesh_volume
            group_fluxes[group_name] = group_mean

            # Create title prefix with energy range
            title_prefix = f"{group_names[group_idx].capitalize()} Group ({energy_bounds[group_idx]:.3e} - {energy_bounds[group_idx+1]:.3e} eV)\n"

            # Create individual group plots
            create_flux_profile_plots(
                group_mean, shape,
                energy_group_dir,  # Use energy_group_dir instead of plot_dir
                f'normalized_flux_profiles_{group_names[group_idx]}.png',
                title_prefix,
                active_core_radius, tank_radius, reflector_radius, half_height
            )

        # Calculate total flux for main plot
        mean_sum = np.sum(mean_data, axis=1)
        flux_mean = mean_sum.reshape(shape) * norm_factor / mesh_volume
    else:
        # No energy bins - direct reshape
        flux_mean = flux.mean.reshape(shape) * norm_factor / mesh_volume
        group_fluxes = None

    # Create main total flux plot
    create_flux_profile_plots(
        flux_mean, shape,
        plot_dir, 'normalized_flux_profiles.png',
        "Total ",
        active_core_radius, tank_radius, reflector_radius, half_height,
        group_fluxes=group_fluxes
    )
