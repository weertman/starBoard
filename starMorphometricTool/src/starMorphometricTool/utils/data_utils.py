import numpy as np
import json
import os
import logging
import math

def convert_numpy_types(obj):
    """Convert NumPy types to native Python for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj


def load_morphometrics(file_path):
    """
    Load morphometric data from a JSON file.

    Args:
        file_path (str): Path to the morphometrics.json file

    Returns:
        dict: The loaded morphometric data, or None if the file couldn't be loaded

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Morphometrics file not found: {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        return data

    except json.JSONDecodeError as e:
        logging.error(f"Error parsing morphometrics JSON: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading morphometrics file: {e}")
        raise


def get_arm_lengths(morphometrics):
    """
    Extract just the arm lengths from morphometrics data.

    Args:
        morphometrics (dict): Morphometrics data

    Returns:
        list: List of arm lengths in mm
    """
    arm_data = morphometrics.get('arm_data', [])
    arm_lengths = [arm[3] for arm in arm_data]  # arm[3] is the length_mm value
    return arm_lengths


def calculate_arm_diameters(morphometrics_data):
    """
    Calculate arm diameters by finding directly opposed arm tips.
    For each arm, it finds the most directly opposite arm (closest to 180° away)
    and calculates the distance between their tips.

    Args:
        morphometrics_data (dict): Morphometrics data containing 'arm_data'

    Returns:
        list: List of tip-to-tip diameter lengths in mm

    Notes:
        - Arm data format is expected to be: [arm_number, x_vec, y_vec, length_mm]
        - Returns empty list if fewer than 2 arms
    """
    arm_data = morphometrics_data.get('arm_data', [])

    # Need at least 2 arms to calculate diameters
    if len(arm_data) < 2:
        return []

    # Calculate angle of each arm vector (in radians)
    arm_angles = []
    for arm in arm_data:
        arm_number, x_vec, y_vec, length_mm = arm
        angle = math.atan2(y_vec, x_vec)
        arm_angles.append((arm_number, angle, x_vec, y_vec, length_mm))

    diameters = []

    # For each arm, find the most opposite arm
    for i, (arm1_number, arm1_angle, x1, y1, length1) in enumerate(arm_angles):
        # Calculate opposite angle (add 180° = π radians)
        opposite_angle = (arm1_angle + math.pi) % (2 * math.pi)

        # Find the arm with angle closest to the opposite
        min_diff = float('inf')
        opposite_arm_idx = -1

        for j, (arm2_number, arm2_angle, _, _, _) in enumerate(arm_angles):
            if j == i:  # Skip self
                continue

            # Calculate angular difference (accounting for circular nature)
            diff = abs((arm2_angle - opposite_angle + math.pi) % (2 * math.pi) - math.pi)

            if diff < min_diff:
                min_diff = diff
                opposite_arm_idx = j

        if opposite_arm_idx != -1:
            _, _, x2, y2, _ = arm_angles[opposite_arm_idx]

            # Calculate diameter (tip-to-tip distance)
            diameter = math.sqrt((x1 + x2) ** 2 + (y1 + y2) ** 2)
            diameters.append(diameter)

    return diameters


# =============================================================================
# Registry Management Functions
# =============================================================================

def get_default_registry():
    """
    Return a default empty registry structure.
    
    Returns:
        dict: Default registry with empty locations, gallery, query, and user_initials sections
    """
    return {
        "locations": [],
        "user_initials": [],
        "gallery": {},
        "query": {}
    }


def load_registry(registry_path):
    """
    Load the registry from a JSON file, or create a new one if it doesn't exist.
    
    Args:
        registry_path (str): Path to the registry.json file
        
    Returns:
        dict: The registry data
    """
    try:
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            # Ensure all required keys exist
            if "locations" not in registry:
                registry["locations"] = []
            if "user_initials" not in registry:
                registry["user_initials"] = []
            if "gallery" not in registry:
                registry["gallery"] = {}
            if "query" not in registry:
                registry["query"] = {}
            return registry
        else:
            # Create default registry
            return get_default_registry()
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing registry JSON: {e}")
        return get_default_registry()
    except Exception as e:
        logging.error(f"Error loading registry: {e}")
        return get_default_registry()


def save_registry(registry_path, registry_data):
    """
    Save the registry to a JSON file.
    
    Args:
        registry_path (str): Path to save the registry.json file
        registry_data (dict): The registry data to save
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        # Ensure directory exists
        registry_dir = os.path.dirname(registry_path)
        if registry_dir and not os.path.exists(registry_dir):
            os.makedirs(registry_dir, exist_ok=True)
            
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=4)
        return True
    except Exception as e:
        logging.error(f"Error saving registry: {e}")
        return False


def get_gallery_identities(registry):
    """
    Get a list of all gallery identity IDs.
    
    Args:
        registry (dict): The registry data
        
    Returns:
        list: Sorted list of gallery identity IDs
    """
    return sorted(registry.get("gallery", {}).keys())


def get_query_identities(registry):
    """
    Get a list of all query identity IDs.
    
    Args:
        registry (dict): The registry data
        
    Returns:
        list: Sorted list of query identity IDs
    """
    return sorted(registry.get("query", {}).keys())


def get_locations(registry):
    """
    Get a list of all known locations.
    
    Args:
        registry (dict): The registry data
        
    Returns:
        list: Sorted list of location strings
    """
    return sorted(registry.get("locations", []))


def add_location(registry, location):
    """
    Add a new location to the registry if it doesn't already exist.
    
    Args:
        registry (dict): The registry data (modified in place)
        location (str): The location to add
        
    Returns:
        bool: True if location was added, False if it already existed
    """
    if "locations" not in registry:
        registry["locations"] = []
    
    location = location.strip()
    if location and location not in registry["locations"]:
        registry["locations"].append(location)
        registry["locations"].sort()
        return True
    return False


def get_user_initials(registry):
    """
    Get a list of all known user initials.
    
    Args:
        registry (dict): The registry data
        
    Returns:
        list: Sorted list of user initials strings
    """
    return sorted(registry.get("user_initials", []))


def add_user_initials(registry, initials):
    """
    Add new user initials to the registry if they don't already exist.
    
    Args:
        registry (dict): The registry data (modified in place)
        initials (str): The initials to add (will be uppercased)
        
    Returns:
        bool: True if initials were added, False if they already existed
    """
    if "user_initials" not in registry:
        registry["user_initials"] = []
    
    initials = initials.strip().upper()
    if initials and initials not in registry["user_initials"]:
        registry["user_initials"].append(initials)
        registry["user_initials"].sort()
        return True
    return False


def add_gallery_identity(registry, identity_id, location, notes=""):
    """
    Add a new gallery identity to the registry.
    
    Args:
        registry (dict): The registry data (modified in place)
        identity_id (str): The identity ID to add
        location (str): The location associated with this identity
        notes (str): Optional notes about the identity
        
    Returns:
        bool: True if identity was added, False if it already existed
    """
    import datetime
    
    if "gallery" not in registry:
        registry["gallery"] = {}
    
    identity_id = identity_id.strip()
    if not identity_id:
        return False
        
    if identity_id not in registry["gallery"]:
        registry["gallery"][identity_id] = {
            "created": datetime.datetime.now().strftime("%Y-%m-%d"),
            "location": location.strip(),
            "notes": notes
        }
        # Also add location to locations list
        add_location(registry, location)
        return True
    return False


def add_query_identity(registry, identity_id, location):
    """
    Add a new query identity to the registry.
    
    Args:
        registry (dict): The registry data (modified in place)
        identity_id (str): The identity ID to add
        location (str): The location associated with this identity
        
    Returns:
        bool: True if identity was added, False if it already existed
    """
    import datetime
    
    if "query" not in registry:
        registry["query"] = {}
    
    identity_id = identity_id.strip()
    if not identity_id:
        return False
        
    if identity_id not in registry["query"]:
        registry["query"][identity_id] = {
            "created": datetime.datetime.now().strftime("%Y-%m-%d"),
            "location": location.strip()
        }
        # Also add location to locations list
        add_location(registry, location)
        return True
    return False


def generate_query_id(registry, location):
    """
    Generate a default query ID using datetime and location.
    Format: YYYY-MM-DD_{location_sanitized}_{sequence}
    
    Args:
        registry (dict): The registry data
        location (str): The location string
        
    Returns:
        str: Generated unique query ID
    """
    import datetime
    
    # Get current date
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Sanitize location for use in filename (replace spaces, special chars)
    location_sanitized = location.strip().replace(" ", "_").replace("/", "-")
    location_sanitized = "".join(c for c in location_sanitized if c.isalnum() or c in "_-")
    
    if not location_sanitized:
        location_sanitized = "unknown"
    
    # Find next sequence number for this date+location combo
    prefix = f"{date_str}_{location_sanitized}_"
    existing_queries = registry.get("query", {})
    
    sequence = 1
    for existing_id in existing_queries.keys():
        if existing_id.startswith(prefix):
            # Extract sequence number
            suffix = existing_id[len(prefix):]
            if suffix.isdigit():
                seq_num = int(suffix)
                if seq_num >= sequence:
                    sequence = seq_num + 1
    
    return f"{prefix}{sequence:03d}"


def identity_exists(registry, identity_type, identity_id):
    """
    Check if an identity already exists in the registry.
    
    Args:
        registry (dict): The registry data
        identity_type (str): Either "gallery" or "query"
        identity_id (str): The identity ID to check
        
    Returns:
        bool: True if identity exists, False otherwise
    """
    if identity_type not in ["gallery", "query"]:
        return False
    return identity_id.strip() in registry.get(identity_type, {})