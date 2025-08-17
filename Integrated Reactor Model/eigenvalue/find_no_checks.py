import xml.etree.ElementTree as ET

def extract_cells(xml_file, target_ids):
    """Extract and print specific cell elements from an OpenMC geometry XML file.

    Parameters
    ----------
    xml_file : str
        Path to the OpenMC geometry XML file
    target_ids : list
        List of cell IDs to extract and analyze

    Returns
    -------
    None

    Notes
    -----
    Prints cells with matching IDs that have material attributes different from "25".
    """
    # Convert target_ids to set for O(1) lookup
    target_ids = set(map(str, target_ids))

    # Parse the XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Find all cell elements and filter those with matching IDs
    for cell in root.findall('.//cell'):
        if cell.get('id') in target_ids:
            # Only print cells where material attribute exists and isn't "25"
            material = cell.get('material')
            if material and material != "25":
                # Convert element to string representation and print it
                cell_str = ET.tostring(cell, encoding='unicode').strip()
                print(cell_str)

# Target cell IDs
target_ids = [13, 26, 39, 52, 65, 78, 91, 104, 117, 132, 145, 160, 173, 186, 199, 212, 225, 238, 251, 264, 277, 290, 303, 316, 329, 342, 355, 368, 381, 394, 407, 420, 433, 446, 460, 473, 486, 501, 514, 527, 540, 553, 566, 579, 592, 605, 618, 631, 634, 638, 641]

extract_cells('eigenvalue/no_checks.xml', target_ids)
