import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SURF_Measurements.surf_channel import SURFChannel
from SURF_Measurements.surf_unit_info import SURFUnitInfo
from SURF_Measurements.surf_trigger import SURFTrigger
from RF_Utils.Pulse import Pulse
from RF_Utils.Waveform import Waveform

from typing import List

class SURFUnit(SURFTrigger):
    """
    Stores SURF unit data. Stores each channel as a SURF channel instance
    """
    def __init__(self, data:np.ndarray|List[SURFChannel], info:SURFUnitInfo|dict, run:int=None, *args, **kwargs):

        self.info:SURFUnitInfo

        if isinstance(info, SURFUnitInfo):
            self.info = SURFUnitInfo(**info.__dict__, **kwargs)
        elif isinstance(info, dict):
            self.info = SURFUnitInfo(**info)

        self.tag = f"SURF : {self.info.surf_unit}{self.info.polarisation} / {self.info.surf_index}" + (", run_"+str(run) if run is not None else "")

        if isinstance(data, np.ndarray):
            self.channels = []
            for rfsoc_channel in range(8):
                self.channels.append(SURFChannel(data=data[self.info.surf_index][rfsoc_channel], info=self.info, rfsoc_channel=rfsoc_channel, run=run))
        elif isinstance(data[0], SURFChannel):
            if all(channel.info.surf_index == data[0].info.surf_index for channel in data):
                self.channels = data
            else:
                raise ValueError("Channels must be from the same SURF Unit")
        else:
            raise TypeError("Data was not inputted in a compatible format.")

        self.beamform_wf:Pulse = None

    def beamform(self, ref_pulse:Pulse = None, window_size = 0.1, min_width=210-15, max_width=210+15, threshold_multiplier=1.8, center_width=5)->Pulse:
        super().beamform(ref_pulse=ref_pulse, window_size=window_size, min_width=min_width, max_width=max_width, threshold_multiplier=threshold_multiplier, center_width=center_width)
        self.beamform_wf.tag = self.tag
        return self.beamform_wf

if __name__ == '__main__':
    from SURF_Measurements.surf_data import SURFData
    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]
    
    run = 23

    filepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_test' / 'mi1a_150.pkl'

    # filepath = parent_dir / 'data' / 'SURF_Data' / 'beamformertrigger' / '72825_beamformertriggertest1_0.pkl'

    # filepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_all_10dboff' / 'mi1a_35.pkl'

    # filepath = parent_dir / 'data' / 'SURF_Data' / '072925_beamformertest1' / f'072925_beamformer_6db_{run}.pkl'

    surf_data = SURFData(filepath=filepath)
    surf_index = 26

    info = {'surf_index' : surf_index}

    surf_unit = SURFUnit(data = surf_data.format_data(), info=info, run=run)
    # surf_unit.plot_unit_grid()

    # fig, ax = plt.subplots()

    surf_unit.plot_antenna_layout()

    surf_unit.plot_beamform()

    # fig, ax = plt.subplots()
    # surf_unit.plot_beamform(ax=ax)
    # ax.set_title(f'SURF : 26/AV, all channel beamform, rftrigger_test/mi1a_35.pkl')



    # surf_unit.plot_beamform_fft()

    # surf_unit.extract_pulse_window(pre=25, post=125)
    # surf_unit.plot_fft_grid(scale = len(surf_unit)/2)


    plt.show()