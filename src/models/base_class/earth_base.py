import numpy as np
import gt4py.cartesian.gtscript as gtscript
import gt4py.storage as gt_storage

from src.constants import DTYPE_ACCURACY

Field3D = gtscript.Field[DTYPE_ACCURACY]

class EarthBase:
    """
    First layer of the earth model.
    The Python implementation of the Earth class to deal with the dunder method, index access, etc...
    There should be no reference to the physical properties of the Earth since it is taken care of in the second layer.
    """
    water_energy: Field3D
    water_mass: Field3D
    air_energy: Field3D
    air_mass: Field3D
    land_energy: Field3D
    land_mass: Field3D
    chunk_mass: Field3D
    heat_transfer_coefficient: Field3D
    specific_heat_capacity: Field3D
    carbon_ppm: Field3D
    backend: str


    def __init__(self, shape: tuple,
                 water_energy: np.ndarray[DTYPE_ACCURACY] = None,
                 water_mass: np.ndarray[DTYPE_ACCURACY] = None,
                 air_energy: np.ndarray[DTYPE_ACCURACY] = None,
                 air_mass: np.ndarray[DTYPE_ACCURACY] = None,
                 land_energy: np.ndarray[DTYPE_ACCURACY] = None,
                 land_mass: np.ndarray[DTYPE_ACCURACY] = None,
                 backend: str = "numpy",
                 parent=None):
        self.shape = shape
        self.parent = parent
        self._total_mass = 0
        self._average_temperature = 0

        if water_energy is None:
            water_energy = np.zeros(shape)
        if water_mass is None:
            water_mass = np.zeros(shape)
        if air_energy is None:
            air_energy = np.zeros(shape)
        if air_mass is None:
            air_mass = np.zeros(shape)
        if land_energy is None:
            land_energy = np.zeros(shape)
        if land_mass is None:
            land_mass = np.zeros(shape)

        self.water_energy = gt_storage.from_array(water_energy, dtype=DTYPE_ACCURACY, backend=backend)
        self.water_mass = gt_storage.from_array(water_mass, dtype=DTYPE_ACCURACY, backend=backend)
        self.air_energy = gt_storage.from_array(air_energy, dtype=DTYPE_ACCURACY, backend=backend)
        self.air_mass = gt_storage.from_array(air_mass, dtype=DTYPE_ACCURACY, backend=backend)
        self.land_energy = gt_storage.from_array(land_energy, dtype=DTYPE_ACCURACY, backend=backend)
        self.land_mass = gt_storage.from_array(land_mass, dtype=DTYPE_ACCURACY, backend=backend)
        self.chunk_mass = gt_storage.empty(self.shape, dtype=DTYPE_ACCURACY, backend=backend)
        self.chunk_temp = gt_storage.empty(self.shape, dtype=DTYPE_ACCURACY, backend=backend)
        self.heat_transfer_coefficient = gt_storage.empty(self.shape, dtype=DTYPE_ACCURACY, backend=backend)
        self.specific_heat_capacity = gt_storage.empty(self.shape, dtype=DTYPE_ACCURACY, backend=backend)
        self.carbon_ppm = gt_storage.empty(self.shape, dtype=DTYPE_ACCURACY, backend=backend)
        self.backend = backend
        
        


    def __len__(self):
        """
        The size of the Grid is always the static size, not the number of elements inside of it
        :return:
        """
        return np.prod(self.shape)
