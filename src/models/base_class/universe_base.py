from typing import Optional

from src.models.physical_class.earth import Earth
from src.models.physical_class.sun import Sun


class UniverseBase:
    """
    First layer of the model Universe.
    Allows iterating over all the objects in the universe.
    """
    earth: Optional[Earth] = None
    sun: Optional[Sun] = None

    def __iter__(self):
        return (x for x in (self.earth, self.sun))
