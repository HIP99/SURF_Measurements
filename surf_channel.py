import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

sys.path.append(os.path.abspath(".."))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SURF_Measurements.surf_channel_info import SURFChannelInfo
from RF_Utils.Pulse import Pulse
from RF_Utils.Waveform import Waveform

from typing import Any

class SURFChannel(SURFChannelInfo):
    """
    Stores SURF channel data. Data needs to be provived. Subclasses could be made to do this automatically

    Assumes the inputted data is the correct 1-d array.
    """
    def __init__(self, data:np.ndarray, surf_channel_name:str = None, surf_index:int = None, channel_index:int = None, run:int=None, sample_frequency:int = 3e9, surf_info: dict[str, Any] = {}, channel_info: dict[str, Any] = {}, *args, **kwargs):
        super().__init__(info=channel_info, surf_info=surf_info, surf_channel_name = surf_channel_name, surf_channel_index = (surf_index, channel_index))

        self.run = run

        tag = f"SURF : {self.surf_channel_name} / {self.surf_index}.{self.rfsoc_channel}" + (", run_"+str(self.run) if self.run is not None else "")
        self.data = Pulse(waveform=data, sample_frequency=sample_frequency, tag=tag)

    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data)
    
    def __array__(self):
        return self.data.waveform

    def plot_data(self, ax: plt.Axes=None, *args, **kwargs):
        """
        Plots data with time axis
        """
        if ax is None:
            fig, ax = plt.subplots()

        self.data.plot_waveform(ax=ax,**kwargs)
        ax.set_ylabel('Raw ADC counts')

    def plot_samples(self, ax: plt.Axes=None, *args, **kwargs):
        """
        Plots data with samples axis
        """
        if ax is None:
            fig, ax = plt.subplots()

        self.data.plot_samples(ax=ax,**kwargs)
        ax.set_ylabel('Raw ADC counts')

    def plot_fft(self, ax: plt.Axes=None, f_start=0, f_stop=2000, log = True, scale = 1.0, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        xf = self.data.xf

        mask = (xf >= f_start*1e6) & (xf <= f_stop*1e6)

        self.data.plot_fft(ax=ax, log = log, mask=mask, scale=scale, **kwargs)

    def extract_pulse_window(self, pre=20, post=120):
        if isinstance(self.data, Pulse):
            self.data.pulse_window(pre=pre, post=post)
        else:
            print("Warning: Data is not a Pulse instance, cannot extract pulse window")

if __name__ == '__main__':
    from SURF_Measurements.surf_data import SURFData

    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    # filepath = parent_dir / 'data' / 'rftrigger_test2' / 'mi2a_35.pkl'
    # filepath = parent_dir / 'data' / 'SURF_Data' / '072925_beamformertest1' / '072925_beamformer_6db_1.pkl'

    filepath = parent_dir / 'data' / 'SURF_Data' / 'beamformertrigger' / '72825_beamformertriggertest1_1.pkl'


    surf_data = SURFData(filepath=filepath)

    surf_index = 26
    channel_index = 4

    surf = SURFChannel(data = surf_data.format_data()[surf_index][channel_index], surf_index=surf_index, channel_index=channel_index)
    
    fig, ax = plt.subplots()


    # surf.extract_pulse_window(100,200)

    surf.plot_samples(ax=ax)


    print(surf.data.detect_energy())
    print(surf.data.hilbert_envelope())
    # fig, ax = plt.subplots()
    # surf.plot_fft(ax=ax, f_start=300, f_stop=1200, log=True)

    plt.legend()
    plt.show()