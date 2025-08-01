import jax
import jax.numpy as jnp
from jax import vmap, jit

from src.models.ABC.celestial_body import CelestialBody
from src.models.base_class.earth_base import EarthBase
import src.constants as constants
from src.constants import DTYPE_ACCURACY

class Earth(EarthBase, CelestialBody):
    """
    Second layer of the Earth model.
    In this layer are all the physical properties and functions of the Earth implemented. It is here that we will add
    new model variables
    """
    albedo: jnp.float16 = 0.3
    CARBON_EMISSIONS_PER_TIME_DELTA: float = 1_000_000  # ppm
    backend: str

    def __init__(self, shape: tuple, radius: float = 6.3781e6, *, parent=None):
        EarthBase.__init__(self, shape, parent=parent)
        CelestialBody.__init__(self,
                               radius)  # The default radius of the earth was found here https://arxiv.org/abs/1510.07674
        self.get_universe().earth = self
        self.get_universe().discover_everything()
        self.origin = (1, 1, 1)

        

        @jit
        def component_ratio(component_mass: jnp.float32, chunk_mass: jnp.float32) -> DTYPE_ACCURACY:
            return component_mass / chunk_mass

        @jit
        def compute_heat_transfer_coefficient(water_mass: jnp.float32, 
                                              air_mass: jnp.float32, 
                                              land_mass: jnp.float32, 
                                              chunk_mass: jnp.float32):

            return component_ratio(water_mass, chunk_mass) * constants.WATER_HEAT_TRANSFER_COEFFICIENT + \
                                            component_ratio(air_mass, chunk_mass) * constants.AIR_HEAT_TRANSFER_COEFFICIENT + \
                                            component_ratio(land_mass, chunk_mass) * constants.LAND_HEAT_TRANSFER_COEFFICIENT
        @jit
        def compute_specific_heat_capacity(water_mass: jnp.float32, 
                                            air_mass: jnp.float32,
                                            land_mass: jnp.float32,
                                            chunk_mass: jnp.float32):

            return component_ratio(water_mass, chunk_mass) * constants.WATER_HEAT_CAPACITY + \
                                        component_ratio(air_mass, chunk_mass) * constants.AIR_HEAT_CAPACITY + \
                                        component_ratio(land_mass, chunk_mass) * constants.LAND_HEAT_CAPACITY
        @jit
        def compute_component_chunk_composition(component_mass: jnp.float32,
                                      chunk_mass: jnp.float32):

            return component_ratio(component_mass, chunk_mass)

        @jit
        def temperature_to_energy(temperature: jnp.float32, mass: jnp.float32):
            """
            Set the temperature of the component by computing the energy from the mass and the temperature
            Water specific is our case (as it is only used to generate a full of water earth)
            :param temperature:
            :return:
            """
            return temperature * mass * constants.WATER_HEAT_CAPACITY
        @jit
        def temperature_to_energy_field(temperature: jnp.float32,
                                        mass: jnp.float32):

            return temperature_to_energy(temperature=temperature, mass=mass)
            

        @jit
        def chunk_temperature(water_energy: jnp.float32, 
                                water_mass: jnp.float32,
                                air_energy: jnp.float32,
                                air_mass: jnp.float32,
                                land_energy: jnp.float32,
                                land_mass: jnp.float32):
            temp = 0.0
            nb_components = 0
            if water_mass != 0:
                temp += water_energy / (constants.WATER_HEAT_CAPACITY * water_mass)
                nb_components += 1
            if air_mass != 0:
                temp += air_energy / (constants.AIR_HEAT_CAPACITY * air_mass)
                nb_components += 1
            if land_mass != 0:
                temp += land_energy / (constants.LAND_HEAT_CAPACITY * land_mass)
                nb_components += 1
            return temp / nb_components

        @jit
        def compute_chunk_temperature(water_energy: jnp.float32, 
                                        water_mass: jnp.float32, 
                                        air_energy: jnp.float32, 
                                        air_mass: jnp.float32, 
                                        land_energy: jnp.float32, 
                                        land_mass: jnp.float32):
            return chunk_temperature(water_energy=water_energy, water_mass=water_mass, air_energy=air_energy, air_mass=air_mass, land_energy=land_energy, land_mass=land_mass)

        @jit
        def compute_chunk_mass(water_mass: jnp.float32,
                                air_mass: jnp.float32, 
                                land_mass: jnp.float32):
            return water_mass + air_mass + land_mass

        @jit
        def sum_vertical_values(in_field: jnp.ndarray):
            return jnp.cumsum(in_field[::-1],axis=0)[::-1]

        @jit
        def add_energy(input_energy: jnp.float32,
                       water_energy: jnp.float32,
                       water_mass: jnp.float32,
                       air_energy: jnp.float32,
                       air_mass: jnp.float32,
                       land_energy: jnp.float32,
                       land_mass: jnp.float32):
            """
            Distribute a same amount of energy on all the chunk of the earth
            """
            chunk_mass = (water_mass + air_mass + land_mass)
            if water_mass != 0:
                water_temp = water_energy + input_energy * (water_mass/chunk_mass)
            else:
                water_temp = water_energy
            if air_mass != 0:
                air_temp = air_energy + input_energy * (air_mass/chunk_mass)
            else:
                air_temp = air_energy
            if land_mass != 0:
                land_temp = land_energy + input_energy * (land_mass/chunk_mass)
            else:
                land_temp = land_energy

            return water_temp, air_temp, land_temp

        self._add_energy = vmap(vmap(vmap(add_energy)))
        self._compute_chunk_mass = vmap(vmap(vmap(compute_chunk_mass)))
        self._compute_chunk_temperature = vmap(vmap(vmap(compute_chunk_temperature)))
        self._sum_vertical_values = sum_vertical_values
        self._temperature_to_energy_field = vmap(vmap(vmap(temperature_to_energy_field)))
        self._compute_heat_transfer_coefficient = vmap(compute_heat_transfer_coefficient)
        self._compute_component_chunk_composition = vmap(vmap(vmap(compute_component_chunk_composition)))
        self._compute_specific_heat_capacity = vmap(vmap(vmap(compute_specific_heat_capacity)))

    @jit
    def sum_horizontal_values(self, field: jnp.ndarray):
        """
        Sum all the values of the input field on K = 0 level
        :param in_field:
        :param out_field:
        :return:
        """    
        return jnp.sum(field[0])

    @property
    def average_temperature(self) -> DTYPE_ACCURACY:
        print("Computing average temperature")
        self.chunk_temp = self._compute_chunk_temperature(self.water_energy, self.water_mass, self.air_energy, self.air_mass, self.land_energy, self.land_mass)
        temp_total_temperature = self._sum_vertical_values(self.chunk_temp)
        self._average_temperature = self.sum_horizontal_values(temp_total_temperature) / len(self)
        
        return self._average_temperature

        

    @property
    def total_mass(self) -> DTYPE_ACCURACY:
        self.chunk_mass = self._compute_chunk_mass(self.water_mass, self.air_mass, self.land_mass)
        temp_total_mass = self._sum_vertical_values(self.chunk_mass)
        self._total_mass = self.sum_horizontal_values(temp_total_mass)
        print("Computing total mass")

        return self._total_mass
    
    @property
    def total_energy(self) -> DTYPE_ACCURACY:
        temp_total_energy = self._compute_chunk_mass(self.water_energy, self.air_energy, self.land_energy)
        temp_sum_energy = self._sum_vertical_values(temp_total_energy)

        return self.sum_horizontal_values(temp_sum_energy)


    @property
    def composition(self):
        composition_mass_dict = dict()
        self.chunk_mass = self._compute_chunk_mass(self.water_mass, self.air_mass, self.land_mass) # If not already computed
        water_composition = self._compute_component_chunk_composition(self.water_mass, self.chunk_mass)
        air_composition = self._compute_component_chunk_composition(self.air_mass, self.chunk_mass)
        land_composition = self._compute_component_chunk_composition(self.land_mass, self.chunk_mass)
        water_composition = self._sum_vertical_values(water_composition)
        air_composition = self._sum_vertical_values(air_composition)
        land_composition = self._sum_vertical_values(land_composition)
        composition_mass_dict["WATER"] = self.sum_horizontal_values(water_composition)/len(self)
        composition_mass_dict["AIR"] = self.sum_horizontal_values(air_composition)/len(self)
        composition_mass_dict["LAND"] = self.sum_horizontal_values(land_composition)/len(self)

        return composition_mass_dict
    

    @property
    def carbon_flux_to_ocean(self):
        """
        The amount of carbon absorbed at every TIME_DELTA by all the ocean
        :return:
        """
        return 100_000

    @property
    def land_carbon_decay(self):
        """
        The amount of carbon released at every TIME_DELTA due to all the biomass decaying
        :return:
        """
        return 330_000

    @property
    def biosphere_carbon_absorption(self):
        """
        The amount of carbon absorbed at every TIME_DELTA due to all the biomass growing
        :return:
        """
        return 300_000

    def __str__(self):
        res = f"Earth : \n" \
              f"- Mass {self.total_mass}\n" \
              f"- Average temperature: {self.average_temperature}\n" \
              f"- Total energy: {self.total_energy}\n" \
              f"- Composition: \n\t{f'{chr(10) + chr(9)} '.join(str(value * 100) + '% ' + key for key, value in self.composition.items())}"
        return res
    
    @jit
    def energy_to_temperature(self, energy: DTYPE_ACCURACY, mass: DTYPE_ACCURACY, heat_capacity: DTYPE_ACCURACY):
        """
        Set the temperature of the component by computing the energy from the mass and the temperature
        :param temperature:
        :return:
        """
        return energy / (mass * heat_capacity)


    def receive_radiation(self, energy: DTYPE_ACCURACY):
        energy = energy * (1 - self.albedo)
        input_energy = energy/len(self)
        input_energy_field = jnp.full(self.shape, input_energy, dtype=DTYPE_ACCURACY)
        self.water_energy, self.air_energy, self.land_energy = self._add_energy(input_energy=input_energy_field,
                                                                                  water_energy=self.water_energy,
                                                                                  water_mass=self.water_mass,
                                                                                  air_energy=self.air_energy,
                                                                                  air_mass=self.air_mass,
                                                                                  land_energy=self.land_energy,
                                                                                  land_mass=self.land_mass)

    def fill_with_water(self):
        """
        Fill the earth with water
        :return:
        """
        key = jax.random.PRNGKey(0) # Keys are not reusable, use slice to make more random for other instances

        self.water_mass = jnp.full(shape=self.shape, fill_value=1000, dtype=DTYPE_ACCURACY)
        water_temp = jax.random.uniform(key, shape=self.shape, dtype=DTYPE_ACCURACY, minval=290, maxval=310)
        self.water_energy = self._temperature_to_energy_field(water_temp, self.water_mass)

        self.chunk_mass = self._compute_chunk_mass(self.water_mass, self.air_mass, self.land_mass)
        self.heat_transfer_coefficient = self._compute_heat_transfer_coefficient(self.water_mass, self.air_mass, self.land_mass, self.chunk_mass)
        self.specific_heat_capacity = self._compute_specific_heat_capacity(self.water_mass, self.air_mass, self.land_mass, self.chunk_mass)
    


    
