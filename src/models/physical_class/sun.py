import math

from src.models.ABC.celestial_body import CelestialBody
from src.models.base_class.sun_base import SunBase


class Sun(SunBase, CelestialBody):
    """
    Second layer of the Sun Model.
    Contains the methods and variable to compute the model of the sun
    """
    def receive_radiation(self, energy: float):
        """
        No behaviour set for the Sun receiving radiation
        :param energy:
        :return:
        """
        return

    def __init__(self, total_energy: float = math.inf, energy_radiated_per_second: float = 3.8e26,
                 radius: float = 6.957e8):
        self.total_energy = total_energy
        self.energy_radiated_per_second = energy_radiated_per_second
        CelestialBody.__init__(self, radius)
        self.get_universe().sun = self
        self.get_universe().discover_everything()

    def __str__(self):
        res = f"Sun :\n"
        if self.total_energy != math.inf:
            res += f"- Total energy remaining {self.total_energy} J\n"
        else:
            res += f"- Infinite amount of energy\n"
        res += f"- Radiating {self.energy_radiated_per_second} W outwards\n"
        if self.get_universe().earth:
            res += f"- Of which {100 * self.solid_angle(self.get_universe().earth) / (4 * math.pi)} % will reach the earth"
        return res
