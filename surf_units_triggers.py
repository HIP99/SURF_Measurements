import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SURF_Measurements.surf_data import SURFData
from SURF_Measurements.surf_units import SURFUnits
from SURF_Measurements.surf_unit_triggers import SURFUnitTriggers
from SURF_Measurements.surf_channel_triggers import SURFChannelTriggers
from SURF_Measurements.surf_triggers import SURFTriggers
from RF_Utils.Pulse import Pulse
from RF_Utils.Waveform import Waveform
from SURF_Measurements.surf_unit_info import SURFUnitInfo

from typing import List

class SURFUnitsTriggers(SURFTriggers):
    """
    Multiple here refers to multiple triggers. Bewarned this might get big.

    This inherits from SURF_Unit_Multiple simply due to shared methods. Do not call the constructor

    Assumes basepath / filename_{run instance}.pkl
    """
    def __init__(self, info:List[SURFUnitInfo]|dict[str,List], basepath:str, filename:str, length:int|tuple=1, data:List[SURFUnits] = None, *args, **kwargs):
        
        self.info:List[SURFUnitInfo]

        if isinstance(info, list):
            temp_info = info
        elif isinstance(info, dict):
            temp_info = []
            for values in zip(*info.values()):
                unique_channel_info = dict(zip(info.keys(), values))
                channel_info = SURFUnitInfo(**unique_channel_info)
                temp_info.append(channel_info)

        super().__init__(basepath=basepath, filename=filename, length=length, data=data, info=temp_info)

        self.tag = f"Test : {filename}, SURF's : {', '.join(str(unit.surf_index) for unit in self.info)}"

    def populate_triggers(self):
        super().populate_triggers(surf_type = SURFUnits)

    def add_trigger(self, run:int):
        super().add_trigger(surf_type = SURFUnits, run=run)

    def add_trigger(self, run:int):
        super().add_trigger(surf_type = SURFUnits, run=run)

    def add_surfs(self, surf_indices:tuple):
        pass

    @property
    def channels_triggers(self) -> List[SURFChannelTriggers]:
        return[
            SURFChannelTriggers(data=list(channel_trigger), info=channel_trigger[0].info)
            for channel_trigger in zip(*[unit_trigger.channels for unit_trigger in self.triggers])
        ]

    @property
    def units_triggers(self) -> List[SURFUnitTriggers]:

        unit_triggers:List[SURFUnitTriggers] = []
        unit_triggers_arr = list(map(list, zip(*self.triggers)))

        for unit_arr in unit_triggers_arr:
            unit_triggers.append(SURFUnitTriggers(data=unit_arr))

        return unit_triggers

    def beamform_units(self) -> List[Pulse]:
        unit_beamforms:List[Pulse] = []

        unit_triggers = self.units_triggers

        for unit in unit_triggers:
            unit_beamforms.append(unit.beamform())

        return unit_beamforms

    def plot_average_beamform(self, ax: plt.Axes=None, omit_list:list = []):
        if ax is None:
            fig, ax = plt.subplots()
        beamform = self.beamform(omit_list=omit_list)
        beamform.plot_waveform(ax = ax)
        ax.set_ylabel('Time (ns)')
        ax.set_ylabel('Raw ADC counts')
        surfs = [unit.surf_index for unit in self.surf_info]
        ax.set_title(f'SURF {surfs}, {self.length} runs Beamform')

if __name__ == '__main__':

    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    basepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_test' 

    filename = 'mi1a'

    surf_indices = [25,26]

    info = {'surf_index':surf_indices}

    surf_triggers = SURFUnitsTriggers(basepath=basepath, filename=filename, length=5, info=info)

    surf_triggers.plot_channel_beamform()
    plt.show()