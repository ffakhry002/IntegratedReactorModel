"""Functions for plotting flux maps."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from inputs import inputs
import matplotlib.patches as patches
from eigenvalue.tallies.normalization import calc_norm_factor
import openmc

def create_flux_map_plots(flux_mean, flux_std, shape, plot_dir, filename, title_prefix="",
                         active_core_radius=None, tank_radius=None, reflector_radius=None, half_height=None):
    """Helper function to create the standard 6-panel flux map plot.

    Parameters
    ----------
    flux_mean : numpy.ndarray
        Mean flux values in shape [nx, ny, nz]
    flux_std : numpy.ndarray
        Standard deviation values in shape [nx, ny, nz]
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
    """
    # Calculate axially averaged flux
    flux_mean_axial_avg = np.mean(flux_mean, axis=0)  # Average along Z axis
    flux_std_axial_avg = np.sqrt(np.mean(flux_std**2, axis=0))  # Propagate uncertainties

    # Get mid-plane indices
    mid_xy = shape[2] // 2  # Z midplane
    mid_xz = shape[1] // 2  # Y midplane

    # Create figure with subplots (3 rows, 2 columns)
    fig, ((ax1, ax1e), (ax2, ax2e), (ax3, ax3e)) = plt.subplots(3, 2, figsize=(24, 24))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # First row: Axially averaged XY flux map and relative error
    im1 = ax1.imshow(flux_mean_axial_avg.T,
                     norm=LogNorm(vmin=np.max(flux_mean)*1e-2, vmax=np.max(flux_mean)),
                     extent=[-reflector_radius, reflector_radius,
                            -reflector_radius, reflector_radius],
                     origin='lower',
                     cmap='viridis')

    # Add boundaries to axially averaged plots (both flux and error)
    for ax in [ax1, ax1e]:
        active_core = patches.Circle((0, 0), active_core_radius, fill=False,
                                   color='red', linestyle='--', label='Active Core')
        tank = patches.Circle((0, 0), tank_radius, fill=False,
                            color='blue', linestyle='--', label='Tank')
        reflector = patches.Circle((0, 0), reflector_radius, fill=False,
                                 color='green', linestyle='--', label='Reflector')
        ax.add_patch(active_core)
        ax.add_patch(tank)
        ax.add_patch(reflector)
        ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)

    ax1.set_title(f'{title_prefix}XY Flux Map (Axially Averaged)')
    ax1.set_xlabel('X [cm]')
    ax1.set_ylabel('Y [cm]')
    plt.colorbar(im1, ax=ax1, label='Flux [n/cm²/s]',
                fraction=0.046, pad=0.04, aspect=30)

    # Axially averaged relative error
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err_avg = flux_std_axial_avg.T / flux_mean_axial_avg.T * 100
        rel_err_avg = np.nan_to_num(rel_err_avg, nan=0.0, posinf=0.0, neginf=0.0)

    im1e = ax1e.imshow(rel_err_avg,
                       extent=[-reflector_radius, reflector_radius,
                              -reflector_radius, reflector_radius],
                       origin='lower',
                       cmap='coolwarm')
    ax1e.set_title(f'{title_prefix}XY Relative Error (Axially Averaged) [%]')
    ax1e.set_xlabel('X [cm]')
    ax1e.set_ylabel('Y [cm]')
    plt.colorbar(im1e, ax=ax1e, fraction=0.046, pad=0.04, aspect=30)

    # Second row: XY flux map at midplane
    im2 = ax2.imshow(flux_mean[mid_xz, :, :].T,
                     norm=LogNorm(vmin=np.max(flux_mean)*1e-2, vmax=np.max(flux_mean)),
                     extent=[-reflector_radius, reflector_radius,
                            -reflector_radius, reflector_radius],
                     origin='lower',
                     cmap='viridis')

    # Add boundaries to midplane plots
    for ax in [ax2, ax2e]:
        active_core = patches.Circle((0, 0), active_core_radius, fill=False,
                                   color='red', linestyle='--', label='Active Core')
        tank = patches.Circle((0, 0), tank_radius, fill=False,
                            color='blue', linestyle='--', label='Tank')
        reflector = patches.Circle((0, 0), reflector_radius, fill=False,
                                 color='green', linestyle='--', label='Reflector')
        ax.add_patch(active_core)
        ax.add_patch(tank)
        ax.add_patch(reflector)
        ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)

    ax2.set_title(f'{title_prefix}XY Flux Map (Z Mid-plane)')
    ax2.set_xlabel('X [cm]')
    ax2.set_ylabel('Y [cm]')
    plt.colorbar(im2, ax=ax2, label='Flux [n/cm²/s]',
                fraction=0.046, pad=0.04, aspect=30)

    # Midplane relative error
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err = flux_std[mid_xz, :, :].T / flux_mean[mid_xz, :, :].T * 100
        rel_err = np.nan_to_num(rel_err, nan=0.0, posinf=0.0, neginf=0.0)

    im2e = ax2e.imshow(rel_err,
                       extent=[-reflector_radius, reflector_radius,
                              -reflector_radius, reflector_radius],
                       origin='lower',
                       cmap='coolwarm')
    ax2e.set_title(f'{title_prefix}XY Relative Error (Z Mid-plane) [%]')
    ax2e.set_xlabel('X [cm]')
    ax2e.set_ylabel('Y [cm]')
    plt.colorbar(im2e, ax=ax2e, fraction=0.046, pad=0.04, aspect=30)

    # Third row: XZ flux map
    flux_xz = flux_mean[:, mid_xy, :]
    im3 = ax3.imshow(flux_xz,
                     norm=LogNorm(),
                     extent=[-reflector_radius, reflector_radius, -half_height, half_height],
                     origin='lower',
                     cmap='viridis')
    ax3.set_title(f'{title_prefix}XZ Flux Map (Y Mid-plane)')
    ax3.set_xlabel('X [cm]')
    ax3.set_ylabel('Z [cm]')
    plt.colorbar(im3, ax=ax3, label='Flux [n/cm²/s]',
                fraction=0.046, pad=0.04, aspect=30)

    # XZ relative error
    flux_xz_std = flux_std[:, mid_xy, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err_xz = flux_xz_std / flux_xz * 100
        rel_err_xz = np.nan_to_num(rel_err_xz, nan=0.0, posinf=0.0, neginf=0.0)

    im3e = ax3e.imshow(rel_err_xz,
                       extent=[-reflector_radius, reflector_radius, -half_height, half_height],
                       origin='lower',
                       cmap='coolwarm')
    ax3e.set_title(f'{title_prefix}XZ Relative Error (Y Mid-plane) [%]')
    ax3e.set_xlabel('X [cm]')
    ax3e.set_ylabel('Z [cm]')
    plt.colorbar(im3e, ax=ax3e, fraction=0.046, pad=0.04, aspect=30)

    # Draw rectangles for XZ plots
    for ax in [ax3, ax3e]:
        active_core_rect = patches.Rectangle((-active_core_radius, -half_height),
                                           2*active_core_radius,
                                           2*half_height,
                                           fill=False, color='red', linestyle='--',
                                           label='Active Core', alpha=0.5)
        tank_rect = patches.Rectangle((-tank_radius, -half_height),
                                    2*tank_radius,
                                    2*half_height,
                                    fill=False, color='blue', linestyle='--',
                                    label='Tank', alpha=0.5)
        reflector_rect = patches.Rectangle((-reflector_radius, -half_height),
                                         2*reflector_radius,
                                         2*half_height,
                                         fill=False, color='green', linestyle='--',
                                         label='Reflector', alpha=0.5)
        ax.add_patch(active_core_rect)
        ax.add_patch(tank_rect)
        ax.add_patch(reflector_rect)
        ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=3)

    plt.savefig(os.path.join(plot_dir, filename),
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_flux_maps(sp, plot_dir, inputs_dict=None):
    """Plot flux maps from the mesh tallies.

    Parameters
    ----------
    sp : openmc.StatePoint
        StatePoint file containing the tally results
    plot_dir : str
        Directory to save the plots
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        from inputs import inputs
        inputs_dict = inputs

    # Get the mesh tally
    mesh_tally = sp.get_tally(name='flux_mesh')
    flux = mesh_tally.get_slice(scores=['flux'])

    # Get dimensions from inputs (in cm)
    core_layout = np.array(inputs_dict['core_lattice'])
    max_assemblies = max([len(row) - list(row).count('C') for row in core_layout])
    if inputs_dict['assembly_type'] == 'Plate':
        assembly_unit_width = (inputs_dict['fuel_plate_width'] + 2*inputs_dict['clad_structure_width']) * 100  # cm
    else:  # Pin type
        assembly_unit_width = inputs_dict['pin_pitch'] * inputs_dict['n_side_pins'] * 100  # cm
    total_assembly_width = max_assemblies * assembly_unit_width
    active_core_radius = total_assembly_width / 2  # Active core (fuel assemblies) radius
    tank_radius = inputs_dict['tank_radius'] * 100  # Tank outer boundary
    reflector_radius = tank_radius + inputs_dict['reflector_thickness'] * 100  # Reflector outer boundary
    half_height = inputs_dict['fuel_height'] * 50  # Just use fuel height

    # Get mesh dimensions from the tally
    mesh_filter = mesh_tally.find_filter(openmc.MeshFilter)
    shape = mesh_filter.mesh.dimension

    # Calculate mesh volume
    dx = 2 * reflector_radius / shape[0]  # Use reflector radius for full width
    dy = 2 * reflector_radius / shape[1]
    dz = 2 * half_height / shape[2]  # Use fuel height only
    mesh_volume = dx * dy * dz

    # Create numpy arrays from tally data and normalize
    power_mw = inputs_dict.get('core_power', 1.0)
    norm_factor = calc_norm_factor(power_mw, sp)

    if inputs_dict['Core_Three_Group_Energy_Bins']:
        # Create directory for energy group plots in the passed plot_dir
        energy_group_dir = os.path.join(plot_dir, '3_energy_group_maps')
        os.makedirs(energy_group_dir, exist_ok=True)

        # Get energy boundaries
        energy_bounds = [0.0, inputs_dict['thermal_cutoff'], inputs_dict['fast_cutoff'], 20.0e6]
        group_names = ['thermal', 'epithermal', 'fast']

        # Process each energy group
        n_spatial = np.prod(shape)
        mean_data = flux.mean.reshape(n_spatial, -1)
        std_data = flux.std_dev.reshape(n_spatial, -1)

        # Plot each energy group
        for group_idx in range(3):
            # Extract and reshape data for this group
            group_mean = mean_data[:, group_idx].reshape(shape) * norm_factor / mesh_volume
            group_std = std_data[:, group_idx].reshape(shape) * norm_factor / mesh_volume

            # Create title prefix with energy range
            title_prefix = f"{group_names[group_idx].capitalize()} Group ({energy_bounds[group_idx]:.3e} - {energy_bounds[group_idx+1]:.3e} eV)\n"

            # Create plots for this energy group
            create_flux_map_plots(
                group_mean, group_std, shape,
                energy_group_dir,
                f'flux_maps_{group_names[group_idx]}.png',
                title_prefix,
                active_core_radius, tank_radius, reflector_radius, half_height
            )

        # Calculate total flux for main plot
        mean_sum = np.sum(mean_data, axis=1)
        std_sum = np.sqrt(np.sum(std_data**2, axis=1))
        flux_mean = mean_sum.reshape(shape) * norm_factor / mesh_volume
        flux_std = std_sum.reshape(shape) * norm_factor / mesh_volume
    else:
        # No energy bins - direct reshape
        flux_mean = flux.mean.reshape(shape) * norm_factor / mesh_volume
        flux_std = flux.std_dev.reshape(shape) * norm_factor / mesh_volume

    # Create main total flux plot
    create_flux_map_plots(
        flux_mean, flux_std, shape,
        plot_dir, 'flux_maps.png',
        "Total ",
        active_core_radius, tank_radius, reflector_radius, half_height
    )
