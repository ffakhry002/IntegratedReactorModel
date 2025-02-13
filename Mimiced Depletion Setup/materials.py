import openmc
import numpy as np
import sys
import os

# Add parent directory to Python path to access inputs.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inputs import inputs

def make_materials(mat_list=None):
    # Convert temperatures from Celsius to Kelvin
    T_fuel_K = inputs['T_fuel']
    T_clad_K = inputs['T_clad']
    T_cool_K = inputs['T_cool']

    material_list = []

    # Create fuel based on selected type
    fuel = openmc.Material(name=inputs['fuel_type'])
    if inputs['fuel_type'] == 'UO2':
        # Create enriched uranium and oxygen for mixing
        u_enriched = openmc.Material(name='u_enriched')
        u_enriched.add_element('U', 1.0, enrichment=inputs['n%'])
        oxygen = openmc.Material(name='O')
        oxygen.add_element('O', 1.0)

        # Mix to create UO2
        fuel = openmc.Material.mix_materials([u_enriched, oxygen],
                                           [0.88146, 0.11854], 'wo',
                                           name='UO2')
        fuel.set_density('g/cm3', 10.3)
        fuel.temperature = T_fuel_K
    elif inputs['fuel_type'] == 'U3Si2':
        fuel.set_density('g/cm3', 12.2)
        fuel.add_element('U', 3.0, enrichment=inputs['n%'])
        fuel.add_element('Si', 2.0)
    elif inputs['fuel_type'] == 'U10Mo':
        fuel.set_density('g/cm3', 17.0)
        fuel.add_element('U', 0.90, enrichment=inputs['n%'])
        fuel.add_element('Mo', 0.10)
    else:
        raise ValueError(f"Unknown fuel type: {inputs['fuel_type']}")

    fuel.depletable = True
    material_list.append(fuel)

    # Create cladding based on selected type
    clad = openmc.Material(name=inputs['clad_type'])
    if inputs['clad_type'] == 'Zircaloy':
        clad.set_density('g/cm3', 6.55)
        clad.add_element('Sn', 0.014, 'wo')
        clad.add_element('Fe', 0.002, 'wo')
        clad.add_element('Cr', 0.001, 'wo')
        clad.add_element('Zr', 0.983, 'wo')
    elif inputs['clad_type'] == 'Al6061':
        clad.set_density('g/cm3', 2.7)
        clad.add_element('Fe', 0.0035, 'wo')
        clad.add_element('Mg', 0.01, 'wo')
        clad.add_element('Si', 0.006, 'wo')
        clad.add_element('Al', 0.9805, 'wo')
    else:
        raise ValueError(f"Unknown clad type: {inputs['clad_type']}")

    clad.temperature = T_clad_K
    material_list.append(clad)

    # Create helium gap
    helium = openmc.Material(name='Helium')
    helium.add_element('He', 1.0)
    helium.set_density('g/cm3', 0.0001785)
    helium.temperature = T_clad_K
    material_list.append(helium)

    # Create water moderator
    water = openmc.Material(name='Light Water')
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cm3', 0.98699)
    water.add_s_alpha_beta('c_H_in_H2O')
    water.temperature = T_cool_K
    material_list.append(water)

    # Create materials collection and dictionary
    materials = openmc.Materials(material_list)
    mat_dict = {mat.name: mat for mat in materials}

    return mat_dict, materials

if __name__ == "__main__":
    mat_dict, materials = make_materials()
