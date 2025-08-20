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

    # Helium gap (for regular fuel assemblies)
    if (mat_list is None) or ('Helium' in mat_list):
        helium = openmc.Material(name='Helium')
        helium.add_element('He', 1.0)

        # # Set temperature based on assembly type
        # if inputs_dict['assembly_type'] == 'Pin':
        #     helium_temp = np.mean(TH_data['T_gap_z'])
        # else:  # Plate fuel doesn't have a gap
        #     helium_temp = default_T
        helium_temp = default_T
        # Calculate density using ideal gas law for fuel assembly gap
        # ρ = PM/RT where P=pressure (Pa), M=molar mass (kg/mol), R=gas constant (J/mol·K), T=temperature (K)
        helium_pressure = 101325  # 1 atm in Pa
        helium_molar_mass = 4.0026e-3  # kg/mol
        gas_constant = 8.314  # J/(mol·K)
        helium_density = (helium_pressure * helium_molar_mass) / (gas_constant * helium_temp)

        helium.set_density('g/cm3', helium_density / 1000)  # Convert kg/m³ to g/cm³
        helium.temperature = helium_temp
        material_list.append(helium)

    #############################################
    # COOLANT MATERIALS
    #############################################

    # Water materials based on coolant type
    coolant_type = inputs_dict.get('coolant_type', 'Light Water')

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

    # Graphite (with S(α,β) table)
    if (mat_list is None) or ('graphite' in mat_list):
        graphite = openmc.Material(name='graphite')
        graphite.add_element('C', 1.0)
        graphite.set_density('g/cm3', 1.75)  # From MCNP materials
        graphite.add_s_alpha_beta('c_Graphite')
        graphite.temperature = default_T
        material_list.append(graphite)

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

    #############################################
    # HTWL IRRADIATION EXPERIMENT MATERIALS
    #############################################

    if (mat_list is None) or ('Titanium' in mat_list):
        # Create titanium component
        titanium = openmc.Material(name='Titanium')
        titanium.add_element('Ti', 1.0)
        titanium.set_density('g/cm3', 4.506)  # Ti density at room temp
        titanium.temperature = 573.15  # 300°C
        material_list.append(titanium)

    # Silicon Carbide (SiC) - Material 601 from HTWL
    if (mat_list is None) or ('SiC' in mat_list):
        sic = openmc.Material(name='SiC')
        sic.add_element('Si', 0.5, percent_type='ao')
        sic.add_element('C', 0.5, percent_type='ao')
        sic.set_density('g/cm3', 3.21)
        sic.temperature = 573.15  # 300°C
        material_list.append(sic)

    # CO2 Gas - Material 81 from HTWL
    if (mat_list is None) or ('CO2' in mat_list):
        co2 = openmc.Material(name='CO2')
        co2.add_nuclide('C12', 1.0)
        co2.add_nuclide('O16', 2.0)

        # Use MCNP reference density (at ~9°C), not operating temperature
        # MCNP convention for consistency
        co2.set_density('g/cm3', 1.9e-3)

        # Set temperature to autoclave operating temperature
        co2.temperature = 373.15  # 100°C
        material_list.append(co2)

    # High-pressure Borated Water (Material 4000 from HTWL)
    # 1400 ppm boron + lithium traces at 300°C
    if (mat_list is None) or ('HP_Borated_Water' in mat_list):
        from CoolProp.CoolProp import PropsSI

        # High pressure water conditions
        hp_water_temp = 573.15  # 300°C in Kelvin
        hp_water_pressure = 15.5e6  # High pressure (15.5 MPa)

        # Get pure water density at high pressure conditions using CoolProp
        hp_water_density = PropsSI('D', 'T', hp_water_temp, 'P', hp_water_pressure, 'Water')  # kg/m³

        hp_borated_water = openmc.Material(name='HP_Borated_Water')
        # From MCNP material 4000: exact atomic fractions
        hp_borated_water.add_nuclide('H1', 0.66667, percent_type='ao')
        hp_borated_water.add_nuclide('O16', 0.33333, percent_type='ao')
        hp_borated_water.add_nuclide('B10', 0.000280, percent_type='ao')
        hp_borated_water.add_nuclide('B11', 0.001120, percent_type='ao')
        hp_borated_water.add_nuclide('Li6', 1.67e-07, percent_type='ao')

        # Calculate composite density (water + 1400 ppm boron)
        rho_boron = 2.34  # g/cm³
        w_water = 0.9986
        w_boron = 0.0014
        V_water = w_water / (hp_water_density / 1000)  # volume per unit mass
        V_boron = w_boron / rho_boron
        rho_composite = 1 / (V_water + V_boron)

        hp_borated_water.set_density('g/cm3', rho_composite)
        hp_borated_water.temperature = hp_water_temp
        material_list.append(hp_borated_water)

    # BWR Fluid (pure water at BWR operating conditions)
    if (mat_list is None) or ('BWR_fluid' in mat_list):
        from CoolProp.CoolProp import PropsSI

        # BWR operating conditions
        bwr_temp = 573.15  # 300°C in Kelvin
        bwr_pressure = 7.2e6  # 7.2 MPa (typical BWR pressure)

        # Get water density at BWR conditions using CoolProp
        water_density_bwr = PropsSI('D', 'T', bwr_temp, 'P', bwr_pressure, 'Water')  # kg/m³

        # Create pure water (no boron for BWR)
        bwr_fluid = openmc.Material(name='BWR_fluid')
        bwr_fluid.add_nuclide('H1', 2.0, percent_type='ao')
        bwr_fluid.add_nuclide('O16', 1.0, percent_type='ao')

        # Set density from CoolProp
        bwr_fluid.set_density('g/cm3', water_density_bwr / 1000)
        bwr_fluid.add_s_alpha_beta('c_H_in_H2O')
        bwr_fluid.temperature = bwr_temp
        material_list.append(bwr_fluid)

    #############################################
    # SIGMA IRRADIATION EXPERIMENT MATERIALS
    #############################################

    # Tungsten - Material 620 from SIGMA
    if (mat_list is None) or ('Tungsten' in mat_list):
        tungsten = openmc.Material(name='Tungsten')
        tungsten.add_nuclide('W182', 0.265, percent_type='ao')
        tungsten.add_nuclide('W183', 0.1431, percent_type='ao')
        tungsten.add_nuclide('W184', 0.3064, percent_type='ao')
        tungsten.add_nuclide('W186', 0.2843, percent_type='ao')
        tungsten.set_density('g/cm3', 19.3)
        tungsten.temperature = 1073.15  # 800°C (sample temperature)
        material_list.append(tungsten)

    # High-temperature Helium for SIGMA
    if (mat_list is None) or ('HT_Helium' in mat_list):
        ht_helium = openmc.Material(name='HT_Helium')
        ht_helium.add_element('He', 1.0)

        # Use MCNP reference density (at ~1°C), not operating temperature
        # MCNP convention for consistency
        ht_helium.set_density('g/cm3', 1.78e-4)

        # Set temperature to operating temperature
        ht_helium.temperature = 1073.15  # 800°C (sample region)
        material_list.append(ht_helium)

    #############################################
    # IRRADIATION LOOP MATERIALS (FULL COMPOSITION)
    #############################################

    # Create materials dictionary for lookup
    materials_dict = {mat.name: mat for mat in material_list}

         # PWR Loop Material - FULL COMPOSITION FROM COMPLEX MODE ANALYSIS
    if (mat_list is None) or ('PWR_loop' in mat_list):
        # Get sample material from inputs
        pwr_sample_name = inputs_dict.get('PWR_sample_fill', 'Vacuum')
        pwr_sample = materials_dict.get(pwr_sample_name)

        if not pwr_sample:
            print(f"Warning: PWR sample material '{pwr_sample_name}' not found. Using Vacuum.")
            pwr_sample = materials_dict['Vacuum']

        # Create materials without S(α,β) tables for mixing
        # Titanium (no S(α,β) table)
        titanium_mix = materials_dict['Titanium']

        # Graphite without S(α,β) table
        graphite_mix = openmc.Material(name='graphite_mix_pwr')
        graphite_mix.add_element('C', 1.0)
        graphite_mix.set_density('g/cm3', 1.75)
        graphite_mix.temperature = 573.15

        # Sample material (already no S(α,β) table for Vacuum)
        sample_mix = pwr_sample

        # HP Borated Water without S(α,β) table
        hp_water_mix = openmc.Material(name='hp_water_mix_pwr')
        hp_water_mix.add_nuclide('H1', 0.66667, percent_type='ao')
        hp_water_mix.add_nuclide('O16', 0.33333, percent_type='ao')
        hp_water_mix.add_nuclide('B10', 0.000280, percent_type='ao')
        hp_water_mix.add_nuclide('B11', 0.001120, percent_type='ao')
        hp_water_mix.add_nuclide('Li6', 1.67e-07, percent_type='ao')

        # Calculate actual PWR water density including boron using CoolProp
        from CoolProp.CoolProp import PropsSI
        pwr_temp = 573.15  # 300°C
        pwr_pressure = 15.5e6  # 15.5 MPa
        pwr_water_density = PropsSI('D', 'T', pwr_temp, 'P', pwr_pressure, 'Water')  # kg/m³
        
        # Account for 1400 ppm boron (same as HP_Borated_Water calculation)
        rho_boron = 2.34  # g/cm³ for elemental boron
        w_water = 0.9986  # mass fraction of water
        w_boron = 0.0014  # mass fraction of boron (1400 ppm)
        
        # Calculate volumes per unit mass
        V_water = w_water / (pwr_water_density / 1000)  # cm³ of water per gram of mixture
        V_boron = w_boron / rho_boron  # cm³ of boron per gram of mixture
        
        # Composite density = 1 / (total volume per unit mass)
        rho_composite = 1 / (V_water + V_boron)  # g/cm³
        
        hp_water_mix.set_density('g/cm3', rho_composite)
        hp_water_mix.temperature = pwr_temp

        # CO2 (no S(α,β) table)
        co2_mix = materials_dict['CO2']

        # Al6061 (no S(α,β) table)
        al6061_mix = materials_dict['Al6061']

        # Light water coolant without S(α,β) table
        coolant_mix = openmc.Material(name='coolant_mix_pwr')
        coolant_mix.add_nuclide('H1', 2.0)
        coolant_mix.add_nuclide('O16', 1.0)
        coolant_mix.set_density('g/cm3', np.mean(TH_data['coolant_density'])/1000)
        coolant_mix.temperature = 573.15

        # Mix using EXACT percentages from complex mode analysis
        pwrloop = openmc.Material.mix_materials(
            [titanium_mix, graphite_mix, sample_mix, hp_water_mix, co2_mix, al6061_mix, coolant_mix],
            [0.166,   # Titanium: 16.6%
            0.021,   # graphite: 2.1%
            0.029,   # Sample: 2.9%
            0.446,   # HP_Borated_Water: 44.6%
            0.097,   # CO2: 9.7%
            0.175,   # Al6061: 17.5%
            0.035],  # Light Water Coolant: 3.5%
            'vo',
            name='PWR_loop'
        )
        pwrloop.temperature = 573.15  # 300°C
        material_list.append(pwrloop)

        # BWR Loop Material - FULL COMPOSITION FROM COMPLEX MODE ANALYSIS
    if (mat_list is None) or ('BWR_loop' in mat_list):
        # Get sample material from inputs
        bwr_sample_name = inputs_dict.get('BWR_sample_fill', 'Vacuum')
        bwr_sample = materials_dict.get(bwr_sample_name)

        if not bwr_sample:
            print(f"Warning: BWR sample material '{bwr_sample_name}' not found. Using Vacuum.")
            bwr_sample = materials_dict['Vacuum']

        # Create materials without S(α,β) tables for mixing
        # Titanium (no S(α,β) table)
        titanium_mix = materials_dict['Titanium']

        # Graphite without S(α,β) table
        graphite_mix = openmc.Material(name='graphite_mix_bwr')
        graphite_mix.add_element('C', 1.0)
        graphite_mix.set_density('g/cm3', 1.75)
        graphite_mix.temperature = 573.15

        # Sample material (already no S(α,β) table for Vacuum)
        sample_mix = bwr_sample

        # BWR fluid without S(α,β) table
        bwr_fluid_mix = openmc.Material(name='bwr_fluid_mix')
        bwr_fluid_mix.add_nuclide('H1', 2.0)
        bwr_fluid_mix.add_nuclide('O16', 1.0)
        # Calculate actual BWR water density using CoolProp (pure water, no boron)
        from CoolProp.CoolProp import PropsSI
        bwr_temp = 573.15  # 300°C
        bwr_pressure = 7.2e6  # 7.2 MPa
        bwr_water_density = PropsSI('D', 'T', bwr_temp, 'P', bwr_pressure, 'Water')  # kg/m³
        bwr_fluid_mix.set_density('g/cm3', bwr_water_density / 1000)  # No boron adjustment needed
        bwr_fluid_mix.temperature = bwr_temp

        # CO2 (no S(α,β) table)
        co2_mix = materials_dict['CO2']

        # Al6061 (no S(α,β) table)
        al6061_mix = materials_dict['Al6061']

        # Light water coolant without S(α,β) table
        coolant_mix = openmc.Material(name='coolant_mix_bwr')
        coolant_mix.add_nuclide('H1', 2.0)
        coolant_mix.add_nuclide('O16', 1.0)
        coolant_mix.set_density('g/cm3', np.mean(TH_data['coolant_density'])/1000)
        coolant_mix.temperature = 573.15

        # Mix using EXACT percentages from complex mode analysis
        bwrloop = openmc.Material.mix_materials(
            [titanium_mix, graphite_mix, sample_mix, bwr_fluid_mix, co2_mix, al6061_mix, coolant_mix],
            [0.166,   # Titanium: 16.6%
            0.021,   # graphite: 2.1%
            0.029,   # Sample: 2.9%
            0.446,   # BWR_fluid: 44.6%
            0.097,   # CO2: 9.7%
            0.175,   # Al6061: 17.5%
            0.035],  # Light Water Coolant: 3.5%
            'vo',
            name='BWR_loop'
        )
        bwrloop.temperature = 573.15  # 300°C
        material_list.append(bwrloop)

        # Gas Capsule Material (SIGMA) - FULL COMPOSITION FROM COMPLEX MODE ANALYSIS
    if (mat_list is None) or ('Gas_capsule' in mat_list):
        # Get sample material from inputs
        gas_sample_name = inputs_dict.get('Gas_capsule_fill', 'Vacuum')
        gas_sample = materials_dict.get(gas_sample_name)

        if not gas_sample:
            print(f"Warning: Gas capsule sample material '{gas_sample_name}' not found. Using Vacuum.")
            gas_sample = materials_dict['Vacuum']

        # Create materials without S(α,β) tables for mixing
        # Titanium (no S(α,β) table)
        titanium_mix = materials_dict['Titanium']

        # HT_Helium (no S(α,β) table)
        ht_helium_mix = materials_dict['HT_Helium']

        # Graphite without S(α,β) table
        graphite_mix = openmc.Material(name='graphite_mix_gas')
        graphite_mix.add_element('C', 1.0)
        graphite_mix.set_density('g/cm3', 1.75)
        graphite_mix.temperature = 1073.15

        # Sample material (already no S(α,β) table for Vacuum)
        sample_mix = gas_sample

        # Light water coolant without S(α,β) table
        coolant_mix = openmc.Material(name='coolant_mix_gas')
        coolant_mix.add_nuclide('H1', 2.0)
        coolant_mix.add_nuclide('O16', 1.0)
        coolant_mix.set_density('g/cm3', np.mean(TH_data['coolant_density'])/1000)
        coolant_mix.temperature = 1073.15

        # Mix using EXACT percentages from complex mode analysis
        gas_capsule = openmc.Material.mix_materials(
            [titanium_mix, ht_helium_mix, graphite_mix, sample_mix, coolant_mix],
            [0.074,   # Titanium: 7.4%
            0.098,   # HT_Helium: 9.8%
            0.543,   # graphite: 54.3%
            0.226,   # Sample: 22.6%
            0.059],  # Light Water Coolant: 5.9%
            'vo',
            name='Gas_capsule'
        )
        gas_capsule.temperature = 1073.15  # 800°C
        material_list.append(gas_capsule)

    materials = openmc.Materials(material_list)
    mat_dict = {mat.name: mat for mat in materials}

    return mat_dict, materials

if __name__ == "__main__":
    mat_dict, materials = make_materials(inputs_dict=inputs)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
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
