import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SURF_Measurements.surf_data import SURFData
from SURF_Measurements.surf_channel import SURFChannel
from SURF_Measurements.surf_channel_info import SURFChannelInfo
from RF_Utils.Pulse import Pulse
from RF_Utils.Waveform import Waveform

from typing import Any

class SURFChannelMultiple(SURFChannelInfo):
    """
    Multiple here refers to multiple triggers, not multiple channels.

    Requires a basepath to find the files, different triggers (runs), and base file name to append run number
    surf_channel_name = AV1
    surf_index = 0-27
    channel_index = 0-7, is the rfsoc channel not the surf channel
    """
    def __init__(self, basepath:str, filename:str, length:int=1, surf_channel_name:str = None, surf_index:int = None, channel_index:int = None, run:int=None, sample_frequency:int = 3e9, surf_info: dict[str, Any] = {}, channel_info: dict[str, Any] = {}, *args, **kwargs):

        super().__init__(info = channel_info, surf_info=surf_info, surf_channel_name = surf_channel_name, surf_channel_index=(surf_index, channel_index))

        self.basepath = basepath
        self.filename = filename
        self.length = length

        self.triggers = []

        """This is just so VSCode will recognise that data points are SURF_Unit instances"""
        filepath = self.basepath / f"{self.filename}_0.pkl"
        surf_data = SURFData(filepath=filepath)
        self.triggers = [SURFChannel(data=surf_data.format_data()[self.surf_index][self.rfsoc_channel], info=self.info, surf_channel_name = self.surf_channel_name, surf_index=self.surf_index, channel_index=self.rfsoc_channel, sample_frequency=sample_frequency)]

        for run in range(1, length):
            filepath = self.basepath / f"{self.filename}_{run}.pkl"
            surf_data = SURFData(filepath=filepath)
            self.triggers.append(SURFChannel(data=surf_data.format_data()[self.surf_index][self.rfsoc_channel], info=self.info, surf_channel_name = self.surf_channel_name, surf_index=self.surf_index, channel_index=self.rfsoc_channel, sample_frequency=sample_frequency))

        self.tag = f"SURF : {self.surf_channel_name} / {self.surf_index}.{self.rfsoc_channel}"

    def __iter__(self):
        return iter(self.triggers)
    
    def __getitem__(self, run):
        return self.triggers[run]
    
    def __len__(self):
        return len(self.triggers)

    def beamform(self):
        beam = Waveform(waveform=self.triggers[0].data.waveform, tag=self.tag+', Beamformed')
        for i in range(1,self.length):
            compare_data = self.triggers[i].data

            corr = np.correlate(beam - beam.mean, compare_data - compare_data.mean, mode='full')
            lags = np.arange(-len(compare_data) + 1, len(beam))
            max_lag = lags[np.argmax(corr)]

            compare_data.correlation_align(beam, max_lag)
            beam.waveform += compare_data
        return beam
    
    def plot_beamform(self, ax: plt.Axes=None, omit_list:list = []):
        if ax is None:
            fig, ax = plt.subplots()
        beamform = self.beamform()
        beamform.plot_waveform(ax = ax)
        ax.set_ylabel('Time (ns)')
        ax.set_ylabel('Raw ADC counts')
        ax.set_title(f'SURF {self.tag} all channel beamform')


if __name__ == '__main__':

    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    basepath = parent_dir / 'data' / 'rftrigger_test'

    filename = 'mi1a'

    surf_index = 26

    channel = SURFChannelMultiple(basepath=basepath, filename=filename, length=5, surf_channel_name='AV2')

    fig, ax = plt.subplots()
    channel.plot_beamform(ax=ax)

    plt.legend()
    plt.show()