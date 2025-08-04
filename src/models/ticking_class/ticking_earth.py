import jax
import jax.numpy as jnp
from jax import jit, vmap

from src.models.ABC.ticking_model import TickingModel
from src.models.physical_class.earth import Earth


class TickingEarth(Earth, TickingModel):
    """
    Third layer of the Earth Model.
    Contains only and all the rules for the model update.

    /!\ Those methods for update must be marked with @TickingModel.on_tick(enabled=True)
    """
    def __init__(self, shape: tuple, radius: float = 6.3781e6, *, parent=None):
        Earth.__init__(self, shape, radius, parent=parent)
        TickingModel.__init__(self)
        self.time_delta = self.get_universe().TIME_DELTA
        self.evaporation_rate = self.get_universe().EVAPORATION_RATE

        def coefficient(heat_transfer_coefficient: jnp.float32,
                             specific_heat_capacity: jnp.float32):
            return heat_transfer_coefficient * specific_heat_capacity * self.time_delta # constant not modified during runtime
        
        def compute_energy_transfer(chunk_temp: jnp.float32, neighbour_k_up: jnp.float32, neighbour_k_down: jnp.float32,
                                    neighbour_y_north: jnp.float32, neighbour_y_south: jnp.float32, neighbour_x_east: jnp.float32,
                                    neighbour_x_west: jnp.float32, coefficient: jnp.float32):
            """
            compute the energy transfer between the grid chunk and its neighbors
            :param grid_chunk:
            :return:
            """
            energy = 0.0
            energy += (neighbour_k_up - chunk_temp) * coefficient
            energy += (neighbour_k_down - chunk_temp) * coefficient
            energy += (neighbour_y_north - chunk_temp) * coefficient
            energy += (neighbour_y_south - chunk_temp) * coefficient
            energy += (neighbour_x_east - chunk_temp) * coefficient
            energy += (neighbour_x_west - chunk_temp) * coefficient
            return energy

        def water_evaporation(water_mass: jnp.float32, air_mass: jnp.float32):
            """
            Evaporate water from the water component of the grid chunk
            :param grid_chunk:
            :return:
            """
            evaporated_mass = self.evaporation_rate * self.time_delta * water_mass # both constants not modified during runtime
            water_temp = water_mass - evaporated_mass
            air_temp = air_mass + evaporated_mass

            return water_temp, air_temp

        def carbon_cycle(carbon_ppm: jnp.float32, carbon_per_chunk: jnp.float32):
            """
            Globally computes carbon flow to be applied to each grid chunk
            :return:
            """
            carbon_temp = carbon_ppm + carbon_per_chunk

            return carbon_temp


        self._water_evaporation = jit(vmap(vmap(vmap(water_evaporation))))
        self._compute_energy_transfer = jit(vmap(vmap(vmap(compute_energy_transfer))))
        self._carbon_cycle = jit(vmap(vmap(vmap(carbon_cycle))))
        self._coefficient = jit(vmap(vmap(vmap(coefficient))))

    @staticmethod
    def neighbour(chunk_temp, shift, axis):
        return jnp.roll(chunk_temp, shift=shift, axis=axis)

    def update(self):
        """
        Special reimplementation of update to update all the components of the earth as well.
        Returns
        -------

        """
        super().update()

    @TickingModel.on_tick(enabled=True)
    def update_temperature(self):
        """
        Update the temperature of each grid chunk
        :return:
        """
        self.chunk_temp = self._compute_chunk_temperature(self.water_energy, self.water_mass, self.air_energy, self.air_mass, self.land_energy, self.land_mass)
        temp_coefficient = self._coefficient(self.heat_transfer_coefficient, self.specific_heat_capacity) #unroll the double loop in compute_energy_transfer
        temp_neighbour_k_up = self.neighbour(self.chunk_temp, shift=-1, axis=0)
        temp_neighbour_k_down = self.neighbour(self.chunk_temp, shift=1, axis=0)
        temp_neighbour_y_north = self.neighbour(self.chunk_temp, shift=-1, axis=1)
        temp_neighbour_y_south = self.neighbour(self.chunk_temp, shift=1, axis=1)
        temp_neighbour_x_east = self.neighbour(self.chunk_temp, shift=-1, axis=2)
        temp_neighbour_x_west = self.neighbour(self.chunk_temp, shift=1, axis=2) # shift for every neighbour and parallelized their computation
        temp_energy = self._compute_energy_transfer(self.chunk_temp, temp_neighbour_k_up, temp_neighbour_k_down,
                                                    temp_neighbour_y_north, temp_neighbour_y_south, temp_neighbour_x_east,
                                                    temp_neighbour_x_west, temp_coefficient) # precise origin for boundaries ?
        self.water_energy, self.air_energy, self.land_energy = self._add_energy(temp_energy, self.water_energy, self.water_mass, self.air_energy, self.air_mass, self.land_energy, self.land_mass)


    @TickingModel.on_tick(enabled=False)
    def water_evaporation(self):
        """
        Evaporate water from the water component of the grid chunk
        :return:
        """
        self.water_mass,  self.air_mass = self._water_evaporation(self.water_mass, self.air_mass)

        

    @TickingModel.on_tick(enabled=False)
    def carbon_cycle(self):
        """
        Globally computes carbon flow to be applied to each grid chunk
        :return:
        """
        carbon_per_chunk = (self.CARBON_EMISSIONS_PER_TIME_DELTA - self.carbon_flux_to_ocean + self.land_carbon_decay - self.biosphere_carbon_absorption) / len(self)
        self.carbon_ppm = self._carbon_cycle(self.carbon_ppm, carbon_per_chunk)