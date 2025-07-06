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
target_ids = [9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 100, 110, 119, 128, 137, 146, 155, 164, 173, 182, 191, 200, 210, 220, 229, 238, 247, 256, 265, 274, 283, 292, 301]

extract_cells('Output/geometry.xml', target_ids)
