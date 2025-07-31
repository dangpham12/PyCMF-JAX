from src.models.ABC.ticking_model import TickingModel
from src.models.physical_class.sun import Sun


class TickingSun(Sun, TickingModel):
    """
    Third layer of the Sun Model.
    Contains only and all the rules for the model update.

    /!\ Those methods for update must be marked with @TickingModel.on_tick(enabled=True)
    """
    def __init__(self):
        Sun.__init__(self)
        TickingModel.__init__(self)

    @TickingModel.on_tick(enabled=True)
    def radiate_energy_outwards(self):
        """
        Update function for the Sun.

        Behavior :
            Updates the amount of energy contained in the sun
            Radiate that energy outward in the universe
        :return:
        """
        energy_per_time_delta = self.energy_radiated_per_second * self.get_universe().TIME_DELTA
        self.get_universe().radiate_inside(energy_per_time_delta, source=self)
