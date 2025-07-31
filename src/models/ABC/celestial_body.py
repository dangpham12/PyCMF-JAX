import jax.numpy as jnp
from abc import abstractmethod
from typing import TYPE_CHECKING

import src.models.physical_class as ph_class

if TYPE_CHECKING:
    from src.models.physical_class.universe import Universe


class CelestialBody:
    """
    Abstract class for celestial bodies. This class must be inherited by any model that has a galactic scale, that is
    that interacts with the universe or other celestial bodies. Examples are the Sun, the Earth, an asteroid, a comet,
    satellites, etc
    """
    __universe: "Universe" = None
    radius: float
    objects_in_line_of_sight: list["CelestialBody"]
    objects_out_of_line_of_sight: list["CelestialBody"]

    def __init__(self, radius: float):
        self.radius = radius
        self.objects_in_line_of_sight = []
        self.objects_out_of_line_of_sight = []

    @staticmethod
    def get_universe() -> "Universe":
        if CelestialBody.__universe is None:
            from src.models.physical_class import universe
            # Reimport exactly the same once at run time when the rest of the program has been built
            CelestialBody.__universe = universe.Universe()
        return CelestialBody.__universe

    @abstractmethod
    def receive_radiation(self, energy: float):
        """
        Defines the behavior of the object when receiving a certain amount of radiation
        :return:
        """

    def solid_angle(self, other: "CelestialBody"):
        """
        Returns the solid angle of the other object as seen from self
        :param other:
        :return: the solid angle in steradians unit
        """
        return jnp.pi * (other.radius ** 2) / (self.get_universe().distance_between(self, other) ** 2)

    def discover(self, other: "CelestialBody"):
        """
        Update the objects_in_line_of_sight list
        :param other:
        :return:
        """
        if isinstance(self, ph_class.earth.Earth) and isinstance(other, ph_class.sun.Sun) or \
                isinstance(self, ph_class.sun.Sun) and isinstance(other, ph_class.earth.Earth):
            if other not in self.objects_in_line_of_sight:
                self.objects_in_line_of_sight.append(other)
            if self not in other.objects_in_line_of_sight:
                other.objects_in_line_of_sight.append(self)
        else:
            if other not in self.objects_out_of_line_of_sight:
                self.objects_out_of_line_of_sight.append(other)
            if self not in other.objects_out_of_line_of_sight:
                other.objects_out_of_line_of_sight.append(self)

    def sees(self, other: "CelestialBody"):
        """
        Check if the two objects see each other or not
        :param other:
        :return:
        """
        if other in self.objects_in_line_of_sight:
            return True
        else:
            if other in self.objects_out_of_line_of_sight:
                return False
            else:
                self.discover(other)
                return self.sees(other)
