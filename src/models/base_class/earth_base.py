import numpy as np

import jax.numpy as jnp
import jax

class EarthBase:
    """
    First layer of the earth model.
    The Python implementation of the Earth class to deal with the dunder method, index access, etc...
    There should be no reference to the physical properties of the Earth since it is taken care of in the second layer.
    """
    water_energy: jnp.ndarray
    water_mass: jnp.ndarray
    air_energy: jnp.ndarray
    air_mass: jnp.ndarray
    land_energy: jnp.ndarray
    land_mass: jnp.ndarray
    chunk_mass: jnp.ndarray
    heat_transfer_coefficient: jnp.ndarray
    specific_heat_capacity: jnp.ndarray
    carbon_ppm: jnp.ndarray

    def __init__(self, shape: tuple,
                 water_energy: jnp.ndarray = None,
                 water_mass: jnp.ndarray = None,
                 air_energy: jnp.ndarray = None,
                 air_mass: jnp.ndarray = None,
                 land_energy: jnp.ndarray = None,
                 land_mass: jnp.ndarray = None,
                 parent=None):
        self.shape = shape
        self.parent = parent
        self._total_mass = 0
        self._average_temperature = 0

        if water_energy is None:
            water_energy = jnp.zeros(shape, dtype=jnp.float32)
        if water_mass is None:
            water_mass = jnp.zeros(shape, dtype=jnp.float32)
        if air_energy is None:
            air_energy = jnp.zeros(shape, dtype=jnp.float32)
        if air_mass is None:
            air_mass = jnp.zeros(shape, dtype=jnp.float32)
        if land_energy is None:
            land_energy = jnp.zeros(shape, dtype=jnp.float32)
        if land_mass is None:
            land_mass = jnp.zeros(shape, dtype=jnp.float32)

        self.water_energy = jnp.array(water_energy)
        self.water_mass = jnp.array(water_mass)
        self.air_energy = jnp.array(air_energy)
        self.air_mass = jnp.array(air_mass)
        self.land_energy = jnp.array(land_energy)
        self.land_mass = jnp.array(land_mass)
        self.chunk_mass = jnp.empty(shape, dtype=jnp.float32) #jnp.empty behaves returns arrays of 0s bc xla compiler cannot create uninitialized arrays
        self.chunk_temp = jnp.empty(shape, dtype=jnp.float32)
        self.heat_transfer_coefficient = jnp.empty(shape, dtype=jnp.float32)
        self.specific_heat_capacity = jnp.empty(shape, dtype=jnp.float32)
        self.carbon_ppm = jnp.empty(shape, dtype=jnp.float32)
        


    def __len__(self):
        """
        The size of the Grid is always the static size, not the number of elements inside of it
        :return:
        """
        return np.prod(self.shape)
