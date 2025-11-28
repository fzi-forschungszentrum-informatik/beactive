""" Utility functions to extract and match metadata from unisens.xml files. """
import glob
import logging
import os
import re
import xml.etree.ElementTree as ET

def match_metadata_from_unisens(unisens_path, criteria):
    """ Function to match metadata with provided criteria."""
    try:
        tree = ET.parse(unisens_path)
        root = tree.getroot()

        # Get namespace
        m = re.match(r'\{(.*)\}', root.tag)
        namespace = m.group(1) if m else ''
        ns = {'u': namespace} if namespace else {}

        # Extract attributes as dictionary
        attrs = {elem.attrib['key']: elem.attrib['value']
                 for elem in root.findall(".//u:customAttribute", ns)}

        # Parse height
        height_str = attrs.get("height")
        try:
            height_value = float(height_str)
            if 1.2 <= height_value <= 2.5:
                height_m = height_value
                height_cm = height_m * 100
            else:
                height_cm = height_value
                height_m = height_cm / 100.0
        except (TypeError, ValueError):
            height_cm = None
            height_m = None

        # Parse weight
        try:
            weight_kg = float(attrs.get("weight"))
        except (TypeError, ValueError):
            weight_kg = None

        # Compute BMI
        try:
            bmi = round(weight_kg / (height_m ** 2), 2) if weight_kg is not None and height_m else None
        except ZeroDivisionError:
            bmi = None

        # Parse age
        try:
            age = int(float(attrs.get("age")))
        except (TypeError, ValueError):
            age = None

        # Gender
        gender = attrs.get("gender")

        # Check criteria
        def check_bound(value, lower_key, upper_key):
            lower = criteria.get(lower_key)
            upper = criteria.get(upper_key)
            # If some bound is given and value is non, bound cannnot be met
            if lower is not None or upper is not None:
                if value is None:
                    return False
            if lower is not None and value < lower:
                return False
            if upper is not None and value > upper:
                return False
            return True

        if not check_bound(height_cm, "height_cm_lower", "height_cm_upper"):
            return False
        if not check_bound(weight_kg, "weight_kg_lower", "weight_kg_upper"):
            return False
        if not check_bound(bmi, "BMI_lower", "BMI_upper"):
            return False
        if not check_bound(age, "age_lower", "age_upper"):
            return False
        if "gender" in criteria and gender != criteria["gender"]:
            return False

        return True

    except Exception as e:
        logging.error(f"Error parsing {unisens_path}: {e}")
        return False


def find_unisens_file(participant_path):
    """Function to traverse participant directories and find unisens.xml path."""
    for path in glob.iglob(os.path.join(participant_path, '**', 'unisens.xml'), recursive=True):
        return path  # Return the first match found
    return None  # Return None if no file is found

def meets_criteria(participant_path, critera):
    """Main function to check the provided critera."""
    unisens_path = find_unisens_file(participant_path)

    if unisens_path is None:
        logging.info("No unisens file found for %s", participant_path)
        return False

    return match_metadata_from_unisens(unisens_path, critera)
