import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SURF_Measurements.surf_data import SURFData
from SURF_Measurements.surf_units import SURFUnits
from SURF_Measurements.surf_unit_multiple import SURFUnitMultiple
from RF_Utils.Pulse import Pulse
from RF_Utils.Waveform import Waveform
from SURF_Measurements.surf_unit_info import SURFUnitInfo

class SURFUnitsMultiple(SURFUnitMultiple):
    """
    Multiple here refers to multiple triggers. Bewarned this might get big.

    This inherits from SURF_Unit_Multiple simply due to shared methods. Do not call the constructor

    Assumes basepath / filename_{run instance}.pkl
    """
    def __init__(self, basepath:str, filename:str, length:int=1, surf_units:list = [], surf_indices:list = [], sample_frequency:int = 3e9, *args, **kwargs):
        self.basepath = basepath
        self.filename = filename
        self.length = length

        self.surf_info = []

        if surf_units:
            for surf_unit in surf_units:
                self.surf_info.append(SURFUnitInfo(surf_unit=surf_unit))
        elif surf_indices:
            for surf_index in surf_indices:
                self.surf_info.append(SURFUnitInfo(surf_index=surf_index))
        else:
            raise ValueError("You have not given a valid range of SURF units to use")

        self.triggers = []

        surf_indices = [unit.surf_index for unit in self.surf_info]

        """This is just so VSCode will recognise that data points are SURF_Units instances"""

        filepath = self.basepath / f"{self.filename}_0.pkl"
        surf_data = SURFData(filepath=filepath)
        self.triggers = [SURFUnits(data=surf_data.format_data(), surf_indices=surf_indices, sample_frequency=sample_frequency)]

        for run in range(1, length):
            filepath = self.basepath / f"{self.filename}_{run}.pkl"
            surf_data = SURFData(filepath=filepath)
            self.triggers.append(SURFUnits(data=surf_data.format_data(), surf_indices=surf_indices, sample_frequency=sample_frequency))

    def plot_average_beamform(self, ax: plt.Axes=None, omit_list:list = []):
        if ax is None:
            fig, ax = plt.subplots()
        beamform = self.average_beamform(omit_list=omit_list)
        beamform.plot_waveform(ax = ax)
        ax.set_ylabel('Time (ns)')
        ax.set_ylabel('Raw ADC counts')
        surfs = [unit.surf_index for unit in self.surf_info]
        ax.set_title(f'SURF {surfs}, {self.length} runs Beamform')

if __name__ == '__main__':

    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    basepath = parent_dir / 'data' / 'rftrigger_test'

    filename = 'mi1a'

    surf_indices = [26]

    surf_triggers = SURFUnitsMultiple(basepath=basepath, filename=filename, length=500, surf_indices=surf_indices)

    surf_triggers.plot_average_beamform()
    plt.show()