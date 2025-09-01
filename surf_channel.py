import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

sys.path.append(os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SURF_Measurements.surf_channel_info import SURFChannelInfo
from SURF_Measurements.surf_unit_info import SURFUnitInfo
from RF_Utils.Pulse import Pulse
from RF_Utils.Waveform import Waveform

from typing import Any

class SURFChannel(Pulse):
    """
    Stores SURF channel data. Data needs to be provived. Subclasses could be made to do this automatically with naming conventions

    Assumes the inputted data is the correct 1-d array.
    """
    def __init__(self, data:np.ndarray|Pulse = None, info:SURFChannelInfo|dict = None, run:int=None, *args, **kwargs):

        if isinstance(info, SURFChannelInfo):
            self.info = info
        elif isinstance(info, SURFUnitInfo):
            self.info = SURFChannelInfo(**info.__dict__, **kwargs)
        elif isinstance(info, dict):
            self.info = SURFChannelInfo(**info)
        else:
            raise TypeError("Infomation input is not in the correct form.")

        self.run = run

        # tag = f"SURF : {self.surf_channel_name} / {self.surf_index}.{self.rfsoc_channel}" + (", run_"+str(self.run) if self.run is not None else "")
        tag = f"SURF : {self.info.surf_channel_name} / {self.info.surf_index}.{self.info.rfsoc_channel}"
        
        if isinstance(data, np.ndarray):
            super().__init__(waveform=data, tag=tag)
        elif isinstance(data, (Pulse, Waveform)):
            self.__dict__.update(data.__dict__)
        else:
            raise ValueError("No Appropriate data included")

    def __str__(self):
        return self.info.__str__() + (f"\nRun : {self.run}\n" if self.run is not None else "")
    
    def __deepcopy__(self, memo):
        """
        I messed up a lil, so this is necessary
        """
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def _match_primer(self, ref_pulse:Pulse = None) -> Pulse:
        if not ref_pulse:
            current_dir = Path(__file__).resolve()
            parent_dir = current_dir.parents[1]

            loaded_list = np.loadtxt(parent_dir / "SURF_Measurements" /"pulse.csv", delimiter=",", dtype=float)
            ref_pulse = Pulse(waveform=np.array(loaded_list))
        return ref_pulse
    
    def plot_data(self, ax: plt.Axes=None, *args, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        super().plot_waveform(ax=ax,**kwargs)
        ax.set_ylabel('Raw ADC counts')

    def plot_samples(self, ax: plt.Axes=None, *args, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        super().plot_samples(ax=ax,**kwargs)
        ax.set_ylabel('Raw ADC counts')

    def plot_fft(self, ax: plt.Axes=None, f_start=0, f_stop=2000, log = True, scale = 1.0, **kwargs):
        super().plot_fft(ax=ax, log = log, f_start=f_start, f_stop=f_stop, scale=scale, **kwargs)

if __name__ == '__main__':
    from SURF_Measurements.surf_data import SURFData

    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    # filepath = parent_dir / 'data' / 'rftrigger_test2' / 'mi2a_35.pkl'
    # filepath = parent_dir / 'data' / 'SURF_Data' / '072925_beamformertest1' / '072925_beamformer_6db_1.pkl'

    filepath = parent_dir / 'data' / 'SURF_Data' / 'beamformertrigger' / '72825_beamformertriggertest1_1.pkl'

    surf_data = SURFData(filepath=filepath)

    surf_index = 25
    rfsoc_channel = 3


    info = {"surf_index":surf_index,"rfsoc_channel":rfsoc_channel}

    surf = SURFChannel(data = surf_data.format_data()[surf_index][rfsoc_channel], info=info)

    fig, ax = plt.subplots()

    surf.plot_samples(ax=ax)

    ax.set_title(surf.info.surf_channel_name)

    plt.legend()
    plt.show()