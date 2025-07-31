import numpy as np
from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import PARALLEL, BACKWARD, computation, interval, IJ, IJK, Field
import gt4py.storage as gt_storage

from src.models.ABC.celestial_body import CelestialBody
from src.models.base_class.earth_base import EarthBase
import src.constants as constants
from src.constants import DTYPE_ACCURACY

Field3D = gtscript.Field[DTYPE_ACCURACY]
backend_opts = dict(
    device_sync=False  # Only available for GPU backends
)

class Earth(EarthBase, CelestialBody):
    """
    Second layer of the Earth model.
    In this layer are all the physical properties and functions of the Earth implemented. It is here that we will add
    new model variables
    """
    albedo: np.float16 = 0.3
    CARBON_EMISSIONS_PER_TIME_DELTA: float = 1_000_000  # ppm
    backend: str

    def __init__(self, shape: tuple, radius: float = 6.3781e6, *, parent=None, backend="numpy"):
        EarthBase.__init__(self, shape, parent=parent, backend=backend)
        CelestialBody.__init__(self,
                               radius)  # The default radius of the earth was found here https://arxiv.org/abs/1510.07674
        self.get_universe().earth = self
        self.get_universe().discover_everything()
        self.backend = backend
        self.origin = (1, 1, 1)

        

        @gtscript.function
        def component_ratio(component_mass: Field3D, chunk_mass: Field3D) -> DTYPE_ACCURACY:
            return component_mass[0, 0, 0] / chunk_mass[0, 0, 0]
        
        def compute_heat_transfer_coefficient(water_mass: Field3D, 
                                              air_mass: Field3D, 
                                              land_mass: Field3D, 
                                              chunk_mass: Field3D,
                                              heat_transfer_coefficient: Field3D):
            with computation(PARALLEL), interval(...):
                heat_transfer_coefficient = component_ratio(water_mass, chunk_mass) * constants.WATER_HEAT_TRANSFER_COEFFICIENT + \
                                             component_ratio(air_mass, chunk_mass) * constants.AIR_HEAT_TRANSFER_COEFFICIENT + \
                                                component_ratio(land_mass, chunk_mass) * constants.LAND_HEAT_TRANSFER_COEFFICIENT
        
        def compute_specific_heat_capacity(water_mass: Field3D, 
                                              air_mass: Field3D, 
                                              land_mass: Field3D, 
                                              chunk_mass: Field3D,
                                              specific_heat_capacity: Field3D):
            with computation(PARALLEL), interval(...):
                specific_heat_capacity = component_ratio(water_mass, chunk_mass) * constants.WATER_HEAT_CAPACITY + \
                                             component_ratio(air_mass, chunk_mass) * constants.AIR_HEAT_CAPACITY + \
                                                component_ratio(land_mass, chunk_mass) * constants.LAND_HEAT_CAPACITY
                
        def compute_chunk_composition(water_mass: Field3D, 
                                      air_mass: Field3D, 
                                      land_mass: Field3D, 
                                      chunk_mass: Field3D,
                                      water_composition: Field3D,
                                      air_composition: Field3D,
                                      land_composition: Field3D):
            with computation(PARALLEL), interval(...):
                water_composition = component_ratio(water_mass, chunk_mass)
                air_composition = component_ratio(air_mass, chunk_mass)
                land_composition = component_ratio(land_mass, chunk_mass)

        @gtscript.function
        def temperature_to_energy(temperature: Field3D, mass: Field3D) -> DTYPE_ACCURACY:
            """
            Set the temperature of the component by computing the energy from the mass and the temperature
            Water specific is our case (as it is only used to generate a full of water earth)
            :param temperature:
            :return:
            """
            return temperature[0, 0, 0] * mass[0, 0, 0] * constants.WATER_HEAT_CAPACITY
        
        def temperature_to_energy_field(temperature: Field3D, mass: Field3D, energy: Field3D):
            with computation(PARALLEL), interval(...):
                energy = temperature_to_energy(temperature=temperature, mass=mass)
            

        @gtscript.function
        def chunk_temperature(water_energy: Field3D, 
            water_mass: Field3D,
            air_energy: Field3D, 
            air_mass: Field3D, 
            land_energy: Field3D, 
            land_mass: Field3D) -> DTYPE_ACCURACY:
            temp = 0.0
            nb_components = 0
            if water_mass[0, 0, 0] != 0:
                temp += water_energy[0, 0, 0] / (constants.WATER_HEAT_CAPACITY * water_mass[0, 0, 0])
                nb_components += 1
            if air_mass[0, 0, 0] != 0:
                temp += air_energy[0, 0, 0] / (constants.AIR_HEAT_CAPACITY * air_mass[0, 0, 0])
                nb_components += 1
            if land_mass[0, 0, 0] != 0:
                temp += land_energy[0, 0, 0] / (constants.LAND_HEAT_CAPACITY * land_mass[0, 0, 0])
                nb_components += 1
            return temp / nb_components

        def compute_chunk_temperature(water_energy: Field3D, 
                                        water_mass: Field3D, 
                                        air_energy: Field3D, 
                                        air_mass: Field3D, 
                                        land_energy: Field3D, 
                                        land_mass: Field3D,
                                        temperature: Field3D) -> DTYPE_ACCURACY:
            with computation(PARALLEL), interval(...):
                temperature = chunk_temperature(water_energy=water_energy, water_mass=water_mass, air_energy=air_energy, air_mass=air_mass, land_energy=land_energy, land_mass=land_mass)

        def compute_chunk_mass(water_mass: Field3D,
                                air_mass: Field3D, 
                                land_mass: Field3D,
                                chunk_mass: Field3D) -> DTYPE_ACCURACY:
            with computation(PARALLEL), interval(...):
                chunk_mass = water_mass[0, 0, 0] + air_mass[0, 0, 0] + land_mass[0, 0, 0]


        def sum_vertical_values(in_field: Field3D,
                           out_field: Field3D):
            """
            Sum all the values of the input field on K dimensions and put the result in the output field at [I, J, 0]
            :param in_field:
            :param out_field:
            :return:
            """
            with computation(PARALLEL), interval(...):
                out_field = in_field[0, 0, 0] # First copy the field
            with computation(BACKWARD), interval(0, -1):
                out_field += out_field[0, 0, 1] # Then add the next element to the previous one
        

        def add_energy(input_energy: Field3D,
                       water_energy: Field3D, 
                       water_mass: Field3D, 
                       air_energy: Field3D, 
                       air_mass: Field3D, 
                       land_energy: Field3D, 
                       land_mass: Field3D):
            """
            Distribute a same amount of energy on all the chunk of the earth
            """
            with computation(PARALLEL), interval(...):
                chunk_mass = (water_mass[0, 0, 0] + air_mass[0, 0, 0] + land_mass[0, 0, 0])
                if water_mass[0, 0, 0] != 0:
                    water_energy[0, 0, 0] += input_energy * (water_mass[0, 0, 0]/chunk_mass)
                if air_mass[0, 0, 0] != 0:
                    air_energy[0, 0, 0] += input_energy * (air_mass[0, 0, 0]/chunk_mass)
                if land_mass[0, 0, 0] != 0:
                    land_energy[0, 0, 0] += input_energy * (land_mass[0, 0, 0]/chunk_mass)

        self._add_energy = gtscript.stencil(definition=add_energy, backend=self.backend, **backend_opts)
        self._compute_chunk_mass = gtscript.stencil(definition=compute_chunk_mass, backend=self.backend, **backend_opts)
        self._compute_chunk_temperature = gtscript.stencil(definition=compute_chunk_temperature, backend=self.backend, **backend_opts)
        self._sum_vertical_values = gtscript.stencil(definition=sum_vertical_values, backend=self.backend, **backend_opts)
        self._temperature_to_energy_field = gtscript.stencil(definition=temperature_to_energy_field, backend=self.backend, **backend_opts)
        self._compute_heat_transfer_coefficient = gtscript.stencil(definition=compute_heat_transfer_coefficient, backend=self.backend, **backend_opts)
        self._compute_chunk_composition = gtscript.stencil(definition=compute_chunk_composition, backend=self.backend, **backend_opts)
        self._compute_specific_heat_capacity = gtscript.stencil(definition=compute_specific_heat_capacity, backend=self.backend, **backend_opts)



    def sum_horizontal_values(self, field: Field3D):
        """
        Sum all the values of the input field on K = 0 level
        :param in_field:
        :param out_field:
        :return:
        """    
        return np.sum(field[:, :, 0])

    @property
    def average_temperature(self) -> DTYPE_ACCURACY:
        print("Computing average temperature")
        self._compute_chunk_temperature(self.water_energy, self.water_mass, self.air_energy, self.air_mass, self.land_energy, self.land_mass, self.chunk_temp)
        temp_total_temperature = gt_storage.empty(self.shape, dtype=DTYPE_ACCURACY, backend=self.backend)
        self._sum_vertical_values(self.chunk_temp, temp_total_temperature)
        self._average_temperature = self.sum_horizontal_values(temp_total_temperature) / len(self)
        
        return self._average_temperature

        

    @property
    def total_mass(self) -> DTYPE_ACCURACY:
        self._compute_chunk_mass(self.water_mass, self.air_mass, self.land_mass, self.chunk_mass)
        temp_total_mass = gt_storage.empty(self.shape, dtype=DTYPE_ACCURACY, backend=self.backend)
        self._sum_vertical_values(self.chunk_mass, temp_total_mass)
        self._total_mass = self.sum_horizontal_values(temp_total_mass)
        print("Computing total mass")
        return self._total_mass
    
    @property
    def total_energy(self) -> DTYPE_ACCURACY:
        temp_total_energy = gt_storage.empty(self.shape, dtype=DTYPE_ACCURACY, backend=self.backend)
        self._compute_chunk_mass(self.water_energy, self.air_energy, self.land_energy, temp_total_energy)
        temp_sum_energy = gt_storage.empty(self.shape, dtype=DTYPE_ACCURACY, backend=self.backend)
        self._sum_vertical_values(temp_total_energy, temp_sum_energy)
        return self.sum_horizontal_values(temp_sum_energy)


    @property
    def composition(self):
        composition_mass_dict = dict()
        self._compute_chunk_mass(self.water_mass, self.air_mass, self.land_mass, self.chunk_mass) # If not already computed
        water_composition = gt_storage.empty(self.shape, dtype=DTYPE_ACCURACY, backend=self.backend)
        air_composition = gt_storage.empty(self.shape, dtype=DTYPE_ACCURACY, backend=self.backend)
        land_composition = gt_storage.empty(self.shape, dtype=DTYPE_ACCURACY, backend=self.backend)
        self._compute_chunk_composition(self.water_mass, self.air_mass, self.land_mass, self.chunk_mass, water_composition, air_composition, land_composition)
        self._sum_vertical_values(water_composition, water_composition)
        self._sum_vertical_values(air_composition, air_composition)
        self._sum_vertical_values(land_composition, land_composition)
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
        input_energy_field = gt_storage.full(self.shape, input_energy, dtype=DTYPE_ACCURACY, backend=self.backend)
        self._add_energy(input_energy=input_energy_field, 
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
        self.water_mass = gt_storage.from_array(np.full(shape=self.shape, fill_value=1000), dtype=DTYPE_ACCURACY, backend=self.backend)
        water_temp = gt_storage.from_array(np.random.uniform(290, 310, self.shape), dtype=DTYPE_ACCURACY, backend=self.backend)
        self._temperature_to_energy_field(water_temp, self.water_mass, self.water_energy)

        self._compute_chunk_mass(self.water_mass, self.air_mass, self.land_mass, self.chunk_mass)
        self._compute_heat_transfer_coefficient(self.water_mass, self.air_mass, self.land_mass, self.chunk_mass, self.heat_transfer_coefficient)
        self._compute_specific_heat_capacity(self.water_mass, self.air_mass, self.land_mass, self.chunk_mass, self.specific_heat_capacity)
    


    
