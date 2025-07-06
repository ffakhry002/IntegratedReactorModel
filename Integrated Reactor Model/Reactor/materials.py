import openmc
import numpy as np
import sys
import os
import pandas as pd

# Add parent directory to Python path to access inputs.py and ThermalHydraulics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_inputs import inputs
from ThermalHydraulics.TH_refactored import THSystem
from ThermalHydraulics.code_architecture.helper_codes.material_properties.coolant_properties import get_coolant_properties

def make_materials(th_system=None, mat_list=None, inputs_dict=None):
    """Create materials for the reactor model.

    Parameters
    ----------
    th_system : THSystem, optional
        THSystem object containing thermal data. If None, creates new one from inputs.
    mat_list : list, optional
        List of specific materials to create. If None, creates all materials.
    inputs_dict : dict, optional
        Custom inputs dictionary. If None, uses the global inputs.

    Returns
    -------
    tuple
        (material_dictionary, materials_collection) where:
        - material_dictionary: Dictionary mapping material names to Material objects
        - materials_collection: OpenMC Materials collection containing all created materials
    """
    # Use provided inputs or default to global inputs
    if inputs_dict is None:
        inputs_dict = inputs

    if th_system is None:
        th_system = THSystem(inputs_dict)
        thermal_state = th_system.calculate_temperature_distribution()

    TH_data = th_system.get_data()
    T_inlet = TH_data['T_inlet']
    default_T = 273.15 + 23

    material_list = []

    #############################################
    # FUELS
    #############################################

    # UO2 Fuel (Standard and Enhanced)
    if (mat_list is None) or ('UO2' in mat_list) or ('UO2-Enhanced' in mat_list):
        # Create standard enrichment fuel
        if 'n%' in inputs_dict:
            enrichment = float(inputs_dict['n%'])
        else:
            if inputs_dict['fuel_type'] == 'UO2':
                raise Exception("must use 'n%' parameter to define enrichment")
            else:
                enrichment = 5  # arbitrary; only get here when plotting

        # Standard enrichment UO2
        u_enriched = openmc.Material(name='u_enriched')
        u_enriched.add_element('U', 1.0, enrichment=enrichment)
        oxygen = openmc.Material(name='O')
        oxygen.add_element('O', 1.0)

        uo2 = openmc.Material.mix_materials([u_enriched, oxygen], [0.88146, 0.11854], 'wo',
                                          name='UO2')
        uo2.set_density('g/cm3', 10.3)
        uo2.temperature = np.mean(TH_data['T_fuel_avg_z'])
        uo2.depletable = True  # Mark as depletable
        material_list.append(uo2)

        # Create enhanced enrichment UO2 if n%E is specified
        if 'n%E' in inputs_dict:
            enrichment_enhanced = float(inputs_dict['n%E'])
            u_enriched_enhanced = openmc.Material(name='u_enriched_enhanced')
            u_enriched_enhanced.add_element('U', 1.0, enrichment=enrichment_enhanced)

            uo2_enhanced = openmc.Material.mix_materials([u_enriched_enhanced, oxygen], [0.88146, 0.11854], 'wo',
                                                       name='UO2-Enhanced')
            uo2_enhanced.set_density('g/cm3', 10.97)
            uo2_enhanced.temperature = np.mean(TH_data['T_fuel_avg_z'])
            uo2_enhanced.depletable = True  # Mark as depletable
            material_list.append(uo2_enhanced)

    # U3Si2 Fuel (Standard and Enhanced)
    if (mat_list is None) or ('U3Si2' in mat_list) or ('U3Si2-Enhanced' in mat_list):
        # Create standard enrichment fuel
        if 'n%' in inputs_dict:
            enrichment = float(inputs_dict['n%'])
        else:
            if inputs_dict['fuel_type'] == 'U3Si2':
                raise Exception("must use 'n%' parameter to define enrichment")
            else:
                enrichment = 19.75  # arbitrary; only get here when plotting

        # Standard enrichment U3Si2
        u_enriched = openmc.Material(name='u_enriched')
        u_enriched.add_element('U', 1.0, enrichment=enrichment)
        si_material = openmc.Material(name='Si')
        si_material.add_element('Si', 1.0)

        u3si2 = openmc.Material.mix_materials([u_enriched, si_material], [0.6, 0.4], 'ao',
                                            name='U3Si2')
        u3si2.set_density('g/cm3', 5.7)
        u3si2.temperature = np.mean(TH_data['T_fuel_avg_z'])
        u3si2.depletable = True  # Mark as depletable
        material_list.append(u3si2)

        # Create enhanced enrichment U3Si2 if n%E is specified
        if 'n%E' in inputs_dict:
            enrichment_enhanced = float(inputs_dict['n%E'])
            u_enriched_enhanced = openmc.Material(name='u_enriched_enhanced')
            u_enriched_enhanced.add_element('U', 1.0, enrichment=enrichment_enhanced)

            u3si2_enhanced = openmc.Material.mix_materials([u_enriched_enhanced, si_material], [0.6, 0.4], 'ao',
                                                         name='U3Si2-Enhanced')
            u3si2_enhanced.set_density('g/cm3', 5.7)
            u3si2_enhanced.temperature = np.mean(TH_data['T_fuel_avg_z'])
            u3si2_enhanced.depletable = True  # Mark as depletable
            material_list.append(u3si2_enhanced)

    # Depleted Uranium
    if (mat_list is None) or ('DU' in mat_list):
        # 379 in PNNL-15870
        du = openmc.Material(name='DU')
        du.add_nuclide('U234', 0.000005, percent_type='ao')
        du.add_nuclide('U235', 0.002532, percent_type='ao')
        du.add_nuclide('U238', 0.997463, percent_type='ao')
        du.set_density('g/cm3', 18.95)
        du.temperature = default_T
        material_list.append(du)

    # U10Mo Fuel (Standard and Enhanced)
    if (mat_list is None) or ('U10Mo' in mat_list) or ('U10Mo-Enhanced' in mat_list):
        # Create standard enrichment fuel
        if 'n%' in inputs_dict:
            enrichment = float(inputs_dict['n%'])
        else:
            if inputs_dict['fuel_type'] == 'U10Mo':
                raise Exception("must use 'n%' parameter to define enrichment")
            else:
                enrichment = 19.75  # arbitrary; only get here when plotting

        # Standard enrichment U10Mo
        u_enriched = openmc.Material(name='u_enriched')
        u_enriched.add_element('U', 0.90, enrichment=enrichment)  # 90 wt% U
        mo = openmc.Material(name='Mo')
        mo.add_element('Mo', 1.0)  # Pure Mo

        u10mo = openmc.Material.mix_materials(
            [u_enriched, mo],
            [0.90, 0.10],  # 90wt% U, 10wt% Mo
            'wo',
            name='U10Mo'
        )
        u10mo.set_density('g/cm3', 17.0)
        u10mo.temperature = np.mean(TH_data['T_fuel_avg_z'])
        u10mo.depletable = True  # Mark as depletable
        material_list.append(u10mo)

        # Create enhanced enrichment U10Mo if n%E is specified
        if 'n%E' in inputs_dict:
            enrichment_enhanced = float(inputs_dict['n%E'])
            u_enriched_enhanced = openmc.Material(name='u_enriched_enhanced')
            u_enriched_enhanced.add_element('U', 0.90, enrichment=enrichment_enhanced)

            u10mo_enhanced = openmc.Material.mix_materials(
                [u_enriched_enhanced, mo],
                [0.90, 0.10],
                'wo',
                name='U10Mo-Enhanced'
            )
            u10mo_enhanced.set_density('g/cm3', 17.0)
            u10mo_enhanced.temperature = np.mean(TH_data['T_fuel_avg_z'])
            u10mo_enhanced.depletable = True  # Mark as depletable
            material_list.append(u10mo_enhanced)

    #############################################
    # CLADDING MATERIALS
    #############################################

    # Zircaloy Cladding (Standard and Enhanced)
    if (mat_list is None) or ('Zircaloy' in mat_list) or ('Zircaloy-Enhanced' in mat_list):
        # Standard cladding
        clad = openmc.Material(name='Zircaloy')
        clad.set_density('g/cm3', 5.77)
        clad.add_nuclide('Zr90', 0.5145)
        clad.add_nuclide('Zr91', 0.1122)
        clad.add_nuclide('Zr92', 0.1715)
        clad.add_nuclide('Zr94', 0.1738)
        clad.add_nuclide('Zr96', 0.0280)
        clad.temperature = np.mean(TH_data['T_clad_middle_z'])
        material_list.append(clad)

        # Enhanced cladding (same composition, different name for coloring)
        clad_enhanced = openmc.Material(name='Zircaloy-Enhanced')
        clad_enhanced.set_density('g/cm3', 5.77)
        clad_enhanced.add_nuclide('Zr90', 0.5145)
        clad_enhanced.add_nuclide('Zr91', 0.1122)
        clad_enhanced.add_nuclide('Zr92', 0.1715)
        clad_enhanced.add_nuclide('Zr94', 0.1738)
        clad_enhanced.add_nuclide('Zr96', 0.0280)
        clad_enhanced.temperature = np.mean(TH_data['T_clad_middle_z'])
        material_list.append(clad_enhanced)

    # HT9 Cladding (Standard and Enhanced)
    if (mat_list is None) or ('HT9' in mat_list) or ('HT9-Enhanced' in mat_list):
        # Standard HT9
        ht9 = openmc.Material(name='HT9')
        ht9_atom_percent = {
            'C': 0.009183,
            'Si': 0.007854,
            'P': 0.000534,
            'S': 0.000344,
            'V': 0.003248,
            'Cr': 0.121971,
            'Mn': 0.006023,
            'Fe': 0.838895,
            'Ni': 0.004698,
            'Mo': 0.005748,
            'W': 0.001500
        }
        for element, ap in ht9_atom_percent.items():
            ht9.add_element(element, ap, percent_type='ao')
        ht9.set_density('g/cm3', 7.874)
        ht9.temperature = np.mean(TH_data['T_clad_middle_z'])
        material_list.append(ht9)

        # Enhanced HT9 (same composition, different name for coloring)
        ht9_enhanced = openmc.Material(name='HT9-Enhanced')
        for element, ap in ht9_atom_percent.items():
            ht9_enhanced.add_element(element, ap, percent_type='ao')
        ht9_enhanced.set_density('g/cm3', 7.874)
        ht9_enhanced.temperature = np.mean(TH_data['T_clad_middle_z'])
        material_list.append(ht9_enhanced)

    # Al6061 Cladding (Standard and Enhanced)
    if (mat_list is None) or ('Al6061' in mat_list) or ('Al6061-Enhanced' in mat_list):
        # Standard Al6061
        al6061 = openmc.Material(name='Al6061')
        al6061.weight_percent = {
            'Fe': 0.35,
            'Zn': 0.125,
            'Cr': 0.195,
            'Mn': 0.075,
            'Mg': 1.000,
            'Si': 0.600,
            'Ti': 0.075,
            'Cu': 0.275,
            'Al': 97.305
        }
        for element, wt in al6061.weight_percent.items():
            al6061.add_element(element, wt, percent_type='wo')
        al6061.set_density('g/cm3', 2.7)
        al6061.temperature = np.mean(TH_data['T_clad_middle_z'])
        material_list.append(al6061)

        # Enhanced Al6061 (same composition, different name for coloring)
        al6061_enhanced = openmc.Material(name='Al6061-Enhanced')
        for element, wt in al6061.weight_percent.items():
            al6061_enhanced.add_element(element, wt, percent_type='wo')
        al6061_enhanced.set_density('g/cm3', 2.7)
        al6061_enhanced.temperature = np.mean(TH_data['T_clad_middle_z'])
        material_list.append(al6061_enhanced)

    # Helium gap
    if (mat_list is None) or ('Helium' in mat_list):
        helium = openmc.Material(name='Helium')
        helium.add_element('He', 1.0)
        helium.set_density('g/cm3', 0.0001785)  # Density at STP

        # Set temperature based on assembly type
        if inputs_dict['assembly_type'] == 'Pin':
            helium.temperature = np.mean(TH_data['T_gap_z'])
        else:  # Plate fuel doesn't have a gap
            helium.temperature = default_T

        material_list.append(helium)

    #############################################
    # COOLANT MATERIALS
    #############################################

    # Water materials based on coolant type
    coolant_type = inputs_dict.get('coolant_type', 'Light Water')

    # Calculate total plate thickness for plate fuel
    if inputs_dict['assembly_type'] == 'Plate':
        # Total plate thickness is fuel meat plus cladding on both sides
        fuel_plate_thickness = inputs_dict['fuel_meat_thickness'] + 2*inputs_dict['clad_thickness']

        # Calculate total channel volume for plate assembly
        channel_thickness = inputs_dict['fuel_plate_pitch'] - fuel_plate_thickness
        channel_width = inputs_dict['fuel_plate_width']
        channel_height = inputs_dict['fuel_height']
        n_channels = inputs_dict['plates_per_assembly'] + 1  # One more channel than plates

        total_channel_volume = channel_thickness * channel_width * channel_height * n_channels
    else:
        # Pin fuel - calculate pin assembly coolant volume
        pin_area = np.pi * inputs_dict['r_clad_outer']**2
        assembly_area = inputs_dict['pin_pitch']**2 * inputs_dict['n_side_pins']**2
        coolant_area = assembly_area - (inputs_dict['n_side_pins']**2 - inputs_dict['n_guide_tubes']) * pin_area
        total_channel_volume = coolant_area * inputs_dict['fuel_height']

    coolant_config = {
        'Light Water': {
            'h_nuclide': 'H1',
            's_alpha_beta': 'c_H_in_H2O'
        },
        'Heavy Water': {
            'h_nuclide': 'H2',
            's_alpha_beta': 'c_D_in_D2O'
        }
    }

    if coolant_type not in coolant_config:
        raise ValueError(f"Unsupported coolant type: {coolant_type}. Must be one of {list(coolant_config.keys())}")

    config = coolant_config[coolant_type]

    # Coolant material
    if (mat_list is None) or (f'{coolant_type} Coolant' in mat_list):
        coolant = openmc.Material(name=f'{coolant_type} Coolant')
        coolant.add_nuclide(config['h_nuclide'], 2.0)
        coolant.add_nuclide('O16', 1.0)
        coolant.set_density('g/cm3', np.mean(TH_data['coolant_density'])/1000)
        coolant.add_s_alpha_beta(config['s_alpha_beta'])
        coolant.temperature = np.mean(TH_data['T_coolant_z'])
        material_list.append(coolant)

    # Feed material
    if (mat_list is None) or (f'{coolant_type} Feed' in mat_list):
        feed = openmc.Material(name=f'{coolant_type} Feed')
        feed.add_nuclide(config['h_nuclide'], 2.0)
        feed.add_nuclide('O16', 1.0)
        feed_density = TH_data['coolant_density'][0]
        feed.set_density('g/cm3', feed_density/1000)
        feed.add_s_alpha_beta(config['s_alpha_beta'])
        feed.temperature = TH_data['T_inlet']
        material_list.append(feed)

    # Outer material
    if (mat_list is None) or (f'{coolant_type} Outer' in mat_list):
        outer = openmc.Material(name=f'{coolant_type} Outer')
        outer.add_nuclide(config['h_nuclide'], 2.0)
        outer.add_nuclide('O16', 1.0)
        outer_density = TH_data['coolant_density'][0]
        outer.set_density('g/cm3', outer_density/1000)
        outer.add_s_alpha_beta(config['s_alpha_beta'])
        outer.temperature = TH_data['T_inlet']
        material_list.append(outer)

    # Plenum material
    if (mat_list is None) or (f'{coolant_type} Plenum' in mat_list):
        plenum = openmc.Material(name=f'{coolant_type} Plenum')
        plenum.add_nuclide(config['h_nuclide'], 2.0)
        plenum.add_nuclide('O16', 1.0)

        if inputs_dict['assembly_type'] == 'Pin':
            # Use outlet temperature for plenum
            coolant_volume = inputs_dict['num_assemblies']*inputs_dict['fuel_height'] * ((inputs_dict['n_side_pins'] * inputs_dict['pin_pitch'])**2-(inputs_dict['n_side_pins']**2 * np.pi*inputs_dict['r_clad_outer']**2))
            outer_volume = inputs_dict['fuel_height']*(np.pi * inputs_dict['tank_radius']**2) - inputs_dict['num_assemblies']*(inputs_dict['n_side_pins']**2 * inputs_dict['pin_pitch'])
        elif inputs_dict['assembly_type'] == 'Plate':
            coolant_volume = inputs_dict['plates_per_assembly']*inputs_dict['num_assemblies']*inputs_dict['fuel_height']*inputs_dict['fuel_plate_width']*(inputs_dict['fuel_plate_pitch']-inputs_dict['fuel_meat_thickness']-2*inputs_dict['clad_thickness'])
            outer_volume = inputs_dict['fuel_height']*((np.pi * inputs_dict['tank_radius']**2) - inputs_dict['num_assemblies']*(inputs_dict['fuel_plate_width']+2*inputs_dict['clad_structure_width'])**2)
        plenum_volume = coolant_volume + outer_volume
        # plenum_temp = (coolant_volume*TH_data['T_outlet'] + outer_volume*TH_data['T_inlet'])/plenum_volume
        plenum_temp = TH_data['T_inlet']

        # Get density at outlet temperature
        plenum_density = np.interp(plenum_temp, TH_data['T_coolant_z'], TH_data['coolant_density'])
        plenum.set_density('g/cm3', plenum_density/1000)
        plenum.add_s_alpha_beta(config['s_alpha_beta'])
        plenum.temperature = plenum_temp
        material_list.append(plenum)

    #############################################
    # STRUCTURAL MATERIALS
    #############################################

    # Steel
    if (mat_list is None) or ('steel' in mat_list):
        # Material modified from openmc PWR example
        # 1. density set to 7.9, same as RPV steel
        # 2. composition of steel kept as-is, just removed water (openmc will normalize the weight fractions)
        # 3. s_alpha_beta removed. no water.
        steel = openmc.Material(name='Steel')
        steel.set_density('g/cm3', 7.9)
        steel.add_nuclide('Fe54', 0.035620772088, 'wo')
        steel.add_nuclide('Fe56', 0.579805982228, 'wo')
        steel.add_nuclide('Fe57', 0.01362750048, 'wo')
        steel.add_nuclide('Fe58', 0.001848545204, 'wo')
        steel.add_nuclide('Ni58', 0.055298376566, 'wo')
        steel.add_nuclide('Mn55', 0.0182870, 'wo')
        steel.add_nuclide('Cr52', 0.145407678031, 'wo')
        steel.temperature = default_T
        material_list.append(steel)

    # SS316
    if (mat_list is None) or ('SS316' in mat_list):
        ss316 = openmc.Material(name='SS316')
        ss316_atom_percent = {
            'Si': 0.019755,
            'Cr': 0.181400,
            'Mn': 0.020198,
            'Fe': 0.650753,
            'Ni': 0.113436
        }
        for element, ap in ss316_atom_percent.items():
            ss316.add_element(element, ap, percent_type='ao')
        ss316.set_density('g/cm3', 8.0)
        ss316.temperature = np.mean(TH_data['T_clad_middle_z'])
        material_list.append(ss316)

    # Concrete
    if (mat_list is None) or ('Concrete' in mat_list):
        concrete = openmc.Material(name='Concrete')
        concrete_weight_percent = {
            'H': 0.022100,
            'C': 0.002484,
            'O': 0.574930,
            'Na': 0.015208,
            'Mg': 0.001266,
            'Al': 0.019953,
            'Si': 0.304627,
            'K': 0.010045,
            'Ca': 0.042951
        }
        for element, wt in concrete_weight_percent.items():
            concrete.add_element(element, wt, percent_type='wo')
        concrete.set_density('g/cm3', 2.3)
        concrete.temperature = default_T
        material_list.append(concrete)

    # MgO
    if (mat_list is None) or ('mgo' in mat_list):
        mgo = openmc.Material(name='mgo')
        mgo.add_nuclide('Mg24', 0.79)
        mgo.add_nuclide('Mg25', 0.10)
        mgo.add_nuclide('Mg26', 0.11)
        mgo.add_nuclide('O16', 1.33)
        mgo.set_density('g/cm3', 3.58)
        mgo.temperature = default_T
        material_list.append(mgo)

    # Beryllium
    if (mat_list is None) or ('beryllium' in mat_list):
        beryllium = openmc.Material(name='beryllium')
        beryllium.add_element('Be', 1.0)
        beryllium.set_density('g/cm3', 1.85)
        beryllium.temperature = default_T
        material_list.append(beryllium)

    # Beryllium Oxide
    if (mat_list is None) or ('beryllium oxide' in mat_list):
        beryllium_oxide = openmc.Material(name='beryllium oxide')
        beryllium_oxide.add_element('Be', 1.0)
        beryllium_oxide.add_element('O', 1.0)
        beryllium_oxide.set_density('g/cm3', 3.02)
        beryllium_oxide.temperature = default_T
        material_list.append(beryllium_oxide)

    # Irradiation Position Fill (Al6061-Water Mixture)
    if (mat_list is None) or ('Test pos-fill' in mat_list):
        # Create Al6061 component
        al6061 = openmc.Material(name='Al6061 component')
        al6061.weight_percent = {
            'Fe': 0.35,
            'Zn': 0.125,
            'Cr': 0.195,
            'Mn': 0.075,
            'Mg': 1.000,
            'Si': 0.600,
            'Ti': 0.075,
            'Cu': 0.275,
            'Al': 97.305
        }
        for element, wt in al6061.weight_percent.items():
            al6061.add_element(element, wt, percent_type='wo')
        al6061.set_density('g/cm3', 2.7)

        # Create water component
        water = openmc.Material(name='Water component')
        water.add_nuclide('H1', 2.0)
        water.add_nuclide('O16', 1.0)
        water_density, _, _, _ = get_coolant_properties(th_system, np.array([T_inlet]))
        water.set_density('g/cm3', water_density[0] / 1000)
        water.temperature = T_inlet

        # Define volume fractions
        al6061_volume_fraction = 0.65
        water_volume_fraction = 0.35

        # Mix materials using volume fractions
        al6061_water_mix = openmc.Material.mix_materials(
            [al6061, water],
            [al6061_volume_fraction, water_volume_fraction],
            'vo',
            name='Test pos'
        )

        # Calculate the effective density of the mixture
        effective_density = (al6061.density * al6061_volume_fraction +
                           water.density * water_volume_fraction)

        al6061_water_mix.set_density('g/cm3', effective_density)
        al6061_water_mix.temperature = np.mean(TH_data['T_clad_middle_z'])
        material_list.append(al6061_water_mix)

    # Vacuum
    if (mat_list is None) or ('Vacuum' in mat_list):
        vacuum = openmc.Material(name='Vacuum')
        vacuum.add_nuclide('H3', 1.0)
        vacuum.add_nuclide('H2', 1.0)
        vacuum.set_density('g/cm3', 1E-10)
        vacuum.temperature = default_T
        material_list.append(vacuum)

    materials = openmc.Materials(material_list)
    mat_dict = {mat.name: mat for mat in materials}

    return mat_dict, materials

if __name__ == "__main__":
    mat_dict, materials = make_materials(inputs_dict=inputs)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Write materials to text file
    output_file = os.path.join(output_dir, 'materials_info.txt')
    with open(output_file, 'w') as f:
        f.write("\nMaterial Properties\n")
        f.write("==================\n")

        # Write coolant materials properties
        f.write("\nCoolant Materials Properties\n")
        f.write("-------------------------\n")
        coolant_type = inputs.get('coolant_type', 'Light Water')
        for mat in materials:
            if coolant_type in mat.name:
                f.write(f"\n{mat.name}:\n")
                f.write(f"  Density: {mat.density:.6f} g/cm³ ({mat.density * 1000:.2f} kg/m³)\n")
                if hasattr(mat, 'temperature'):
                    f.write(f"  Temperature: {mat.temperature:.2f} K\n")
                if mat._sab:
                    f.write(f"  S(α,β) Tables: {mat._sab}\n")
        f.write("\n")

        # Then write regular material information
        for name, mat in mat_dict.items():
            if coolant_type not in name:  # Skip coolant materials as they were already written
                f.write(f"Material: {name}\n")
                f.write("-" * (len(name) + 10) + "\n")

                # Write density with units
                if hasattr(mat, '_density_units'):
                    f.write(f"Density: {mat.density:.6f} {mat._density_units}\n")
                else:
                    f.write(f"Density: {mat.density:.6f} g/cm³\n")

                # Write temperature if available
                if hasattr(mat, 'temperature'):
                    f.write(f"Temperature: {mat.temperature:.2f} K\n")

                # Write composition
                f.write("Composition:\n")

                # Get atom densities
                atom_densities = mat.get_nuclide_atom_densities()
                if atom_densities:
                    f.write("  Atomic Densities (atoms/b-cm):\n")
                    for nuclide, density in atom_densities.items():
                        f.write(f"    {nuclide}: {density:.6e}\n")

                # Write original composition specifications if available
                if hasattr(mat, 'percent_type'):
                    f.write(f"\n  Original Composition Specification ({mat.percent_type}):\n")
                    if mat.percent_type == 'wo':
                        for nuclide, percent in mat._nuclides.items():
                            f.write(f"    {nuclide}: {percent[1]:.6f} weight fraction\n")
                    elif mat.percent_type == 'ao':
                        for nuclide, percent in mat._nuclides.items():
                            f.write(f"    {nuclide}: {percent[1]:.6f} atom fraction\n")

                # Write element specifications if available
                if hasattr(mat, 'weight_percent'):
                    f.write("\n  Element Weight Percentages:\n")
                    for element, percent in mat.weight_percent.items():
                        f.write(f"    {element}: {percent:.6f}%\n")

                # Write S(α,β) tables if present
                if mat._sab:
                    f.write("\nS(α,β) Tables:\n")
                    for sab in mat._sab:
                        f.write(f"  {sab}\n")

                f.write("\n")

    print(f"\nMaterial information has been written to: {output_file}")
    print("\nCreated Materials:")
    for name in mat_dict:
        print(f"- {name}")
