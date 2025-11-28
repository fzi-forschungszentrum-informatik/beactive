"""Module for creating unique IDs."""

class IdCreator():
    """Class for creating unique IDs."""
    def __init__(self):
        self.id_counter = 0

    def get_id(self):
        """Get a unique ID."""
        id = self.id_counter
        self.id_counter += 1
        return id
