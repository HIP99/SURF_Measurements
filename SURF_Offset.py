import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SURF_Data import SURF_Data
from RF_Utils.Pulse import Pulse

from scipy.signal import find_peaks

class SURF_Offset(SURF_Data):
    def __init__(self, filepath:str, surf_num:int = 14, channel_num = 6, clock_cycle:int=10, *args, **kwargs):
        self.data = None
        self.surf_num = surf_num
        self.channel_num = channel_num
        self.clock_cycle = clock_cycle

        super().__init__(filepath = filepath, *args, **kwargs)

        self.format_data()


    def format_data(self):
        all_data = super().format_data()

        self.data = Pulse(waveform=all_data[self.surf_num][self.channel_num], sample_frequency=3e9, tag = f'{self.surf_num}_{self.channel_num}_{self.clock_cycle}', role = None, offset=None)


    def plot_data(self, ax: plt.Axes=None, *args, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        self.data.plot_waveform(ax=ax,**kwargs)
        # ax.plot(self.data, label = f'{self.surf}_{self.run}', **kwargs)
        # ax.set_xlabel('Sample number')
        ax.set_ylabel('Raw ADC counts')

if __name__ == '__main__':
    from pathlib import Path
    
    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    fig, ax = plt.subplots()

    name = 'maybe'
    surf = 5
    channel = 2

    file_path = parent_dir / 'AMPA_RF_Testing' / 'data' / f'{name}.pkl'
    pckl = SURF_Offset(filepath = file_path, surf_num=surf, channel_num=channel, clock_cycle = 10)

    pckl.plot_data(ax=ax)

    pckl.data.upsampleFreqDomain(factor=3)

    pckl.plot_data(ax=ax)

    print(pckl.data.find_pulse_peak(height=50))

    plt.show()