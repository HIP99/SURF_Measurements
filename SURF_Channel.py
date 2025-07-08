import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SURF_Measurements.SURF_Data import SURF_Data
# from SURF_Measurements.SURF_Data import SURF_Data
from RF_Utils.Pulse import Pulse
from RF_Utils.Waveform import Waveform


class SURF_Channel(SURF_Data):
    """
    Surf data is extracted for everything single surf channel (224 total)
    SURF channel only needs the name of the surf channel and it will extract that surfs data from the whole
    """
    def __init__(self, filepath:str, data = None, surf:str = None, surf_index:int = None, channel_index:int = None, run:int=None, *args, **kwargs):

        self.surf = surf
        self.surf_index = surf_index
        self.channel_index = channel_index
        self.run = run
        self.data = None
        
        super().__init__(filepath = filepath, *args, **kwargs)

        if data is None:
            self.format_data()
        else:
            self.data = Waveform(waveform=data, sample_frequency=3e9, tag = f'SURF_{self.surf_index}_{self.channel_index}'+ (str(self.run) if self.run is not None else ""))


    def __len__(self):
        return len(self.data)

    @property
    def surf_name(self):
        return self.surf[:-1]
    
    @property
    def channel_num(self):
        return int(self.surf[-1])

    def get_surf_index(self):
        self.surf_index = self.surf_mapping.index(self.surf_name)
        self.channel_index = self.channel_mapping[self.channel_num-1]

    
    def format_data(self):
        all_data = super().format_data()

        if self.surf:
            self.get_surf_index()

        ##Maybe should be Pulse idk
        if self.surf:
            self.data = Pulse(waveform=all_data[self.surf_index][self.channel_index], sample_frequency=3e9, tag = f'SURF : {self.surf}_{self.run}')
        else:
            self.data = Waveform(waveform=all_data[self.surf_index][self.channel_index], sample_frequency=3e9, tag = f'SURF_{self.surf_index}_{self.channel_index}'+ (str(self.run) if self.run is not None else ""))

    def plot_data(self, ax: plt.Axes=None, *args, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()


        self.data.plot_waveform(ax=ax,**kwargs)
        ax.set_ylabel('Raw ADC counts')


    def plot_samples(self, ax: plt.Axes=None, *args, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        self.data.plot_samples(ax=ax,**kwargs)
        ax.set_ylabel('Raw ADC counts')


    def plot_fft(self, ax: plt.Axes=None, f_start=0, f_stop=2000, log = True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        xf = self.data.xf

        mask = (xf >= f_start*1e6) & (xf <= f_stop*1e6)

        self.data.plot_fft(ax=ax, log = log, mask=mask, **kwargs)

    def extract_pulse_window(self, pre=20, post=120):
        if isinstance(self.data, Pulse):
            self.data.pulse_window(pre=pre, post=post)

if __name__ == '__main__':
    surf = "IH8"
    run = 0
    surf_index = 5
    channel_index = 2

    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    filepath = parent_dir / 'data' / 'SURF_Data' / f'SURF{surf}' / f'SURF{surf}_{run}.pkl'
    filepath = parent_dir / 'data' / 'Offset_data' / 'idk'
    filepath = parent_dir / 'data' / 'maybe.pkl'

    run0 = SURF_Channel(filepath = filepath, surf=None, surf_index = surf_index, channel_index = channel_index, run=None)

    fig, ax = plt.subplots()

    run0.plot_data(ax=ax)

    plt.legend()
    plt.show()