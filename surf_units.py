import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SURF_Measurements.surf_unit import SURFUnit
from SURF_Measurements.surf_unit_info import SURFUnitInfo
from RF_Utils.Pulse import Pulse
from RF_Utils.Waveform import Waveform

class SURFUnits():
    """
    Stores multiple SURF units for an individual trigger
    """
    def __init__(self, data:np.ndarray, surf_units:list = [], surf_indices:list = [], sample_frequency:int = 3e9, run:int = None, *args, **kwargs):

        self.surf_info = []

        if surf_units:
            for surf_unit in surf_units:
                self.surf_info.append(SURFUnitInfo(surf_unit=surf_unit))
        elif surf_indices:
            for surf_index in surf_indices:
                self.surf_info.append(SURFUnitInfo(surf_index=surf_index))
        else:
            raise ValueError("You have not given a valid range of SURF units to use")

        self.surfs = {}

        for surf in self.surf_info:
            self.surfs[surf['Surf Index']] = SURFUnit(data = data, surf_info=surf.info, surf_index = surf['Surf Index'], sample_frequency = sample_frequency, run = run, *args, **kwargs)

    def __iter__(self):
        return iter(self.surfs.values())
    
    def __getitem__(self, key):
        return self.surfs[key]

    def __len__(self):
        return len(self.surfs)

    def __contains__(self, key):
        ## key in instance
        return key in self.surfs

    def keys(self):
        return self.surfs.keys()

    def values(self):
        return self.surfs.values()

    def items(self):
        return self.surfs.items()
    

    def extract_pulse_window(self, pre=20, post=120):
        for surf in self.values():
            for channel in surf.channels:
                channel.extract_pulse_window(pre=pre, post=post)

    
    def beamform(self, omit_list:list = []):
        """
        Omit list will need to be a list of lists, one for each SURF unit
        Omit is supposed to be channels omitted not surf units
        Do we beamform a Unit then Units or do we beamform channels
        """
        beam = None
        first = True
        for i, surf in enumerate(self.values()):
            for j, channel in enumerate(surf.channels):
                try:
                    if j in omit_list[i]:
                        continue
                except:
                    pass
                if first:
                    beam = Waveform(waveform=channel.data.waveform)
                    first = False
                else:
                    compare_data = channel.data
                    corr = np.correlate(beam - beam.mean, compare_data - compare_data.mean, mode='full')
                    lags = np.arange(-len(compare_data) + 1, len(beam))
                    max_lag = lags[np.argmax(corr)]

                    compare_data.correlation_align(beam, max_lag)
                    beam.waveform += compare_data
        return beam

    def plot_beamform(self, ax: plt.Axes=None, omit_list:list = []):
        if ax is None:
            fig, ax = plt.subplots()
        beamform = self.beamform(omit_list=omit_list)
        beamform.plot_waveform(ax = ax)
        ax.set_ylabel('Time (ns)')
        ax.set_ylabel('Raw ADC counts')
        surfs = [unit.surf_index for unit in self.surf_info]
        ax.set_title(f'SURF(s) {surfs} all channel beamform')

if __name__ == '__main__':
    from SURF_Measurements.surf_data import SURFData
    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    basepath = parent_dir / 'data'

    filepath = basepath / 'rftrigger_test' / 'mi1a_35.pkl'

    surf_data = SURFData(filepath=filepath)

    surf_indices = [5, 26]

    surf_units = SURFUnits(data = surf_data.format_data(), surf_indices=surf_indices)

    surf_units.plot_beamform()
    plt.legend()
    plt.show()