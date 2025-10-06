import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SURF_Measurements.surf_unit import SURFUnit
from SURF_Measurements.surf_unit_info import SURFUnitInfo
from SURF_Measurements.surf_trigger import SURFTrigger
from SURF_Measurements.surf_channel import SURFChannel
from RF_Utils.Pulse import Pulse

from typing import List

class SURFUnits(SURFTrigger):
    """
    Stores multiple SURF units for an individual trigger
    """
    def __init__(self, data:np.ndarray|List[SURFUnit], info:List[SURFUnitInfo]|dict[str,List], run:int = None, *args, **kwargs):

        self.info:List[SURFUnitInfo]
        self.units:List[SURFUnit]

        if isinstance(info, dict):
            self.info = []
            for values in zip(*info.values()):
                unique_unit_info = dict(zip(info.keys(), values))

                unit_info = SURFUnitInfo(**unique_unit_info)
                self.info.append(unit_info)
        elif isinstance(info[0], SURFUnitInfo):
            """
            This will break at this point if the input is wrong. If the input is wrong we'd need to stop anyway
            """
            self.info = info

        if isinstance(data, np.ndarray):
            self.units:List[SURFUnit] = []
            for unit_info in self.info:
                self.units.append(SURFUnit(data=data, info=unit_info))
        elif isinstance(data[0], SURFUnit):
            self.units = data
        else:
            raise TypeError("Data was not inputted in a compatible format.")

    def __iter__(self):
        return iter(self.units)

    def __array__(self):
        return self.units
    
    @property
    def channels(self) -> List[SURFChannel]:
        return [ch for unit in self.units for ch in unit.channels]
    
    def add_unit(self, data, surf_index:int):
        self.units.append(SURFUnit(data = data[surf_index], surf_index=surf_index, run=self.run))

    def remove_channel(self, surf_indices:tuple):
        super().remove_channel(surf_indices)

        for unit in self.units:
            for i, channel in enumerate(unit.channels):
                if (channel.info.surf_index == surf_indices[0] and
                    channel.info.rfsoc_channel == surf_indices[1]):
                    del self.channels[i]
                    break

    def matched_sum_units(self) -> List[Pulse]:
        unit_matched_sums:List[Pulse] = []
        for unit in self.units:
            unit_matched_sum = unit.matched_sum()
            unit_matched_sum.tag = unit.tag
            unit_matched_sums.append(unit_matched_sum)
        return unit_matched_sums
    
    def plot_unit_matched_sum(self):
        unit_matched_sums = self.matched_sum_units()

        cols = len(unit_matched_sums)%3
        rows = int(np.ceil(len(unit_matched_sums) / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*3))

        axes = axes.flatten()

        for i, unit_matched_sum in enumerate(unit_matched_sums):
            unit_matched_sum.plot_samples(ax=axes[i])
            axes[i].set_title(unit_matched_sum.tag)
            axes[i].legend()

    def plot_matched_sum(self, ax: plt.Axes=None):
        if ax is None:
            fig, ax = plt.subplots()
        super().plot_matched_sum(ax)
        ax.set_title(f'SURF(s) {[unit.surf_index for unit in self.info]} all channel matched sum')

if __name__ == '__main__':
    from SURF_Measurements.surf_data import SURFData
    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    basepath = parent_dir / 'data' / 'SURF_Data'

    filepath = basepath / 'rftrigger_test' / 'mi1a_150.pkl'

    surf_data = SURFData(filepath=filepath)

    surf_indices = [25,26]

    info = {'surf_index' : surf_indices}

    surf_units = SURFUnits(data = surf_data.format_data(), info=info)

    surf_units.plot_antenna_layout()
    # plt.legend()
    plt.show()