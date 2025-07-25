import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SURF_Measurements.surf_channel import SURFChannel
from SURF_Measurements.surf_unit_info import SURFUnitInfo
from RF_Utils.Pulse import Pulse
from RF_Utils.Waveform import Waveform

from typing import Any

class SURFUnit(SURFUnitInfo):
    """
    Stores SURF unit data. Stores each channel as a SURF channel instance
    """
    def __init__(self, data:np.ndarray, surf:str = None, surf_index:int = None, run:int=None, sample_frequency:int = 3e9, surf_info: dict[str, Any] = {}, *args, **kwargs):

        super().__init__(info = surf_info, surf_unit = surf, surf_index=surf_index)
        
        self.run = run

        data = data[self.surf_index]

        self.tag = f"SURF : {self.surf_unit}{self.polarisation} / {self.surf_index}" + (", run_"+str(self.run) if self.run is not None else "")

        self.channels = [SURFChannel(data = data[0], surf_info = self.info.copy(), surf_index = self.surf_index, channel_index = 0, run=self.run, sample_frequency = sample_frequency)]

        for i in range(1,8):
            self.channels.append(SURFChannel(data = data[i], surf_info = self.info.copy(), surf_index = self.surf_index, channel_index = i, run=self.run, sample_frequency = sample_frequency))

    def __len__(self):
        return len(self.channels[0])
    
    def __iter__(self):
        return iter(self.channels)
    
    def extract_pulse_window(self, pre=20, post=120):
        for channel in self.channels:
            channel.extract_pulse_window(pre=pre, post=post)

    def beamform(self, omit_list:list = []):
        data = Waveform(waveform=self.channels[0].data.waveform, tag=self.tag+', Beamformed')
        # fig, ax = plt.subplots()
        for i in range(1,8):
            if i in omit_list:
                continue
            compare_data = self.channels[i].data
            corr = np.correlate(data - data.mean, compare_data - compare_data.mean, mode='full')
            lags = np.arange(-len(compare_data) + 1, len(data))
            max_lag = lags[np.argmax(corr)]

            compare_data.correlation_align(data, max_lag)
            data.waveform += compare_data
        return data

    def plot_beamform(self, ax: plt.Axes=None, omit_list:list = []):
        if ax is None:
            fig, ax = plt.subplots()
        beamform = self.beamform()
        beamform.plot_waveform(ax = ax)
        ax.set_ylabel('Time (ns)')
        ax.set_ylabel('Raw ADC counts')
        ax.set_title(f'SURF {self.surf_unit} all channel beamform')

    def plot_beamform_fft(self, ax: plt.Axes=None, omit_list:list = [], f_start=300, f_stop=1200, log = True, scale = 1.0, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        beam = self.beamform()
        xf = beam.xf

        beam.plot_fft_smoothed(ax=ax, log = log, f_start=300, f_stop=1200, scale=scale, **kwargs)

    def plot_unit_series(self, ax: plt.Axes=None):
        if ax is None:
            fig, ax = plt.subplots()

        delay = 0
        for channel in self.channels:
            channel.plot_samples(ax=ax, delay = delay)
            delay+=1024

        ax.set_ylim(-1000, 1000)
        ax.legend()

    def plot_unit_grid(self):
        fig, axs = plt.subplots(4, 2, figsize=(12, 10), sharex=True)

        for i, channel in enumerate(self.channels):
            row = i // 2
            col = i % 2
            channel.plot_samples(ax=axs[row, col])
            axs[row, col].set_title(f"{channel.data.tag}")
            axs[row, col].set_ylim(-1000, 1000)

        plt.tight_layout()

    def plot_fft_grid(self, f_start=300, f_stop=1200, log = True, scale=1.0):
        fig, axs = plt.subplots(4, 2, figsize=(12, 10), sharex=True)

        for i, channel in enumerate(self.channels):
            row = i // 2
            col = i % 2
            channel.plot_fft(ax=axs[row, col], f_start=f_start, f_stop=f_stop, log = log, scale = scale)
            axs[row, col].set_title(f"{channel.data.tag}")

        plt.tight_layout()

if __name__ == '__main__':
    from SURF_Measurements.surf_data import SURFData
    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    # filepath = parent_dir / 'data' / 'rftrigger_test2' / 'mi2a_150.pkl'

    filepath = parent_dir / 'data' / 'rftrigger_test' / 'mi1a_35.pkl'

    # filepath = parent_dir / 'data' / 'rftrigger_all_10dboff' / 'mi1a_35.pkl'


    surf_data = SURFData(filepath=filepath)

    surf_index = 26

    surf_unit = SURFUnit(data = surf_data.format_data(), surf = "AV", surf_index=surf_index)


    # surf_unit.extract_pulse_window()

    fig, ax = plt.subplots()
    # surf_unit.plot_unit_series(ax=ax)
    surf_unit.plot_beamform(ax=ax)
    # ax.set_title(f'SURF : 26/AV, all channel beamform, rftrigger_test/mi1a_35.pkl')



    # surf_unit.plot_beamform_fft()

    # surf_unit.extract_pulse_window(pre=25, post=125)
    # surf_unit.plot_fft_grid(scale = len(surf_unit)/2)


    plt.legend()
    plt.show()