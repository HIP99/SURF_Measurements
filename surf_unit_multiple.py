import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SURF_Measurements.surf_data import SURFData
from SURF_Measurements.surf_unit import SURFUnit
from SURF_Measurements.surf_unit_info import SURFUnitInfo

from typing import Any

class SURFUnitMultiple(SURFUnitInfo):
    """
    Multiple here refers to multiple triggers.

    Requires a basepath to find the files, different triggers (runs), and base file name to append run number
    surf = AV
    surf_index = 0-27
    """
    def __init__(self, basepath:str, filename:str, length:int=1, surf:str = None, surf_index:str = None, sample_frequency:int = 3e9, surf_info: dict[str, Any] = {}, *args, **kwargs):
        super().__init__(info = surf_info, surf_unit = surf, surf_index=surf_index)

        self.basepath = basepath
        self.filename = filename
        self.length = length

        self.triggers = []

        """This is just so VSCode will recognise that data points are SURF_Unit instances"""
        filepath = self.basepath / f"{self.filename}_0.pkl"
        surf_data = SURFData(filepath=filepath)
        self.triggers = [SURFUnit(data=surf_data.format_data(), surf_info=self.info, surf = self.surf_unit, surf_index=self.surf_index, sample_frequency=sample_frequency)]

        for run in range(1, length):
            filepath = self.basepath / f"{self.filename}_{run}.pkl"
            surf_data = SURFData(filepath=filepath)
            self.triggers.append(SURFUnit(data=surf_data.format_data(), surf_info=self.info, surf = self.surf_unit, surf_index=self.surf_index,sample_frequency=sample_frequency))

        self.tag = f"SURF : {self.surf_unit}{self.polarisation} / {self.surf_index}"


    def __iter__(self):
        return iter(self.triggers)
    
    def __getitem__(self, run):
        return self.triggers[run]
    
    def __len__(self):
        return len(self.triggers)

    def average_beamform(self, omit_list:list = []):
        """
        Average beamform across all triggers
        """
        average_beam = self.triggers[0].beamform(omit_list=omit_list)

        for trigger in self.triggers[1:]:
            compare_beam = trigger.beamform(omit_list=omit_list)

            corr = np.correlate(average_beam - average_beam.mean, compare_beam - compare_beam.mean, mode='full')
            lags = np.arange(-len(compare_beam) + 1, len(average_beam))
            max_lag = lags[np.argmax(corr)]

            compare_beam.correlation_align(average_beam, max_lag)

            average_beam.waveform += compare_beam
        average_beam.waveform /= len(self.triggers)
        return average_beam
    
    def plot_average_beamform(self, ax: plt.Axes=None, omit_list:list = []):
        if ax is None:
            fig, ax = plt.subplots()
        beamform = self.average_beamform(omit_list=omit_list)
        beamform.plot_waveform(ax = ax)
        ax.set_ylabel('Time (ns)')
        ax.set_ylabel('Raw ADC counts')
        ax.set_title(f'{self.tag} , {self.length} runs Beamform')
    
if __name__ == '__main__':

    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    basepath = parent_dir / 'data' / 'rftrigger_test'

    filename = 'mi1a'

    surf_index = 26

    surf_triggers = SURFUnitMultiple(basepath=basepath, filename=filename, length=5, surf_index=surf_index)

    surf_triggers.plot_average_beamform()
    plt.show()