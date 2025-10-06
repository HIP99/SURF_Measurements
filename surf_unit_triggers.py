import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SURF_Measurements.surf_data import SURFData
from SURF_Measurements.surf_channel import SURFChannel
from SURF_Measurements.surf_unit import SURFUnit

from SURF_Measurements.surf_triggers import SURFTriggers

from SURF_Measurements.surf_unit_info import SURFUnitInfo

from RF_Utils.Pulse import Pulse

from typing import List

class SURFUnitTriggers(SURFTriggers):
    """
    Multiple here refers to multiple triggers.

    Requires a basepath to find the files, different triggers (runs), and base file name to append run number
    surf = AV
    surf_index = 0-27
    """
    def __init__(self, info:SURFUnitInfo|dict, basepath:str, filename:str, length:int|tuple=1, data:List[SURFUnit] = None, *args, **kwargs):
        
        self.info:SURFUnitInfo
        
        if isinstance(info, SURFUnitInfo):
            info = SURFUnitInfo(**info.__dict__, **kwargs)
        elif isinstance(info, dict):
            info = SURFUnitInfo(**info)

        super().__init__(basepath=basepath, filename=filename, length=length, data=data, info=info)

        self.tag = f"SURF : {self.info.surf_unit}{self.info.polarisation} / {self.info.surf_index}"

    def populate_triggers(self):
        super().populate_triggers(surf_type = SURFUnit)

    def add_trigger(self, run:int, surf_data = None):
        super().add_trigger(surf_type = SURFUnit, run=run, surf_data=surf_data)

    def unit_trigger_matched_sum(self, **kwargs)->List[Pulse]:
        trigger_matched_sums:List[Pulse] = []

        for trigger in self.triggers:
            trigger.matched_sum(**kwargs)
            trigger_matched_sums.append(trigger.matched_sum(**kwargs))

        return trigger_matched_sums
    

if __name__ == '__main__':

    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    basepath = parent_dir / 'data' / 'SURF_Data' / '072925_beamformertest1' 
    filename = '072925_beamformer_6db'

    basepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_test' 
    filename = 'mi1a'


    # basepath = parent_dir / 'data' / 'SURF_Data' / '082625_AGCtest_all' / '082625_AGCtest_0dB'
    # filename = '082625_AGCtest0'

    surf_index = 26

    info = {'surf_index':surf_index}

    surf_triggers = SURFUnitTriggers(basepath=basepath, filename=filename, info=info)

    # channels = surf_triggers.channels
    # arr=[]
    # temp = []
    # for channel_arr in channels:
    #     for channel in channel_arr:
    #         temp.append(channel.info.rfsoc_channel)
    #     arr.append(temp)
    #     temp=[]
    # print(arr)

    surf_triggers.plot_channel_matched_sum()

    plt.show()