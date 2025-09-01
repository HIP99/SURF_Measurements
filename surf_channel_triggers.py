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

from typing import Any, List

class SURFChannelTriggers():
    """
    Multiple here refers to multiple triggers, not multiple channels.

    Requires a basepath to find the files, different triggers (runs), and base file name to append run number
    surf_channel_name = AV1
    surf_index = 0-27
    channel_index = 0-7, is the rfsoc channel not the surf channel
    """
    def __init__(self, basepath:str=None, filename:str=None, length:int=1, data:List[SURFChannel] = None, info:SURFChannelInfo|dict = None, *args, **kwargs):

        if isinstance(info, SURFChannelInfo):
            self.info = info
        elif isinstance(info, dict):
            self.info = SURFChannelInfo(**info)
        else:
            raise TypeError("Infomation input is not in the correct form.")

        self.basepath = basepath
        self.filename = filename
        self.length = length

        self.triggers:List[SURFChannel]

        if data is not None:
            """Are we inputting all the data at once"""
            self.triggers = data
            self.length = len(self.triggers)

        else:
            self.triggers = []
            for run in range(length):
                filepath = self.basepath / f"{self.filename}_{run}.pkl"
                surf_data = SURFData(filepath=filepath)
                self.triggers.append(SURFChannel(data=surf_data.format_data()[self.info.surf_index][self.info.rfsoc_channel],  info=self.info, run=run))
            
            del surf_data

        self.tag = f"SURF : {self.info.surf_channel_name} / {self.info.surf_index}.{self.info.rfsoc_channel}"
        self.beamform_wf:Pulse = None

    def __iter__(self):
        return iter(self.triggers)
    
    def __getitem__(self, run):
        return self.triggers[run]
    
    def __len__(self):
        return len(self.triggers)
    
    def beamform(self, ref_pulse:Pulse = None, window_size = 0.1, min_width=210-15, max_width=210+15, threshold_multiplier=1.8, center_width=5)->Pulse:
        if not ref_pulse:
            current_dir = Path(__file__).resolve()
            parent_dir = current_dir.parents[1]

            loaded_list = np.loadtxt(parent_dir / "SURF_Measurements" /"pulse.csv", delimiter=",", dtype=float)
            ref_pulse = Pulse(waveform=np.array(loaded_list))

        beam = Pulse(waveform=np.zeros(1024), sample_frequency=3e9, tag = self.tag+"_Beamform")

        omitted_triggers = []

        for channel in self.triggers:

            compare_data = channel.copy()

            max_lag = compare_data.match_filter_check(ref_pulse=ref_pulse, window_size=window_size, min_width=min_width, max_width=max_width, threshold_multiplier=threshold_multiplier, center_width=center_width)

            if not max_lag:
                omitted_triggers.append(channel.run)
            else:
                beam.waveform += compare_data.waveform

        print(f"Omitted {len(omitted_triggers)} Triggers\nOmitted Triggers : {omitted_triggers}")
        self.beamform_wf = beam

        return self.beamform_wf
        
    def plot_beamform(self, ax: plt.Axes=None, correlation_strength_coef=4, correlation_threshold = 120, mask = slice(None), **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        
        if not self.beamform_wf:
            self.beamform(correlation_strength_coef, correlation_threshold)
        self.beamform_wf.plot_waveform(ax = ax, mask=mask)
        ax.set_ylabel('Time (ns)')
        ax.set_ylabel('Raw ADC counts')
        ax.set_title(f'{self.tag}, {self.length} runs Beamform')

    def plot_beamform_samples(self, ax: plt.Axes=None, mask = slice(None), **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        if not self.beamform_wf:
            self.beamform()

        self.beamform_wf.plot_samples(ax = ax, mask = mask, **kwargs)
        ax.set_ylabel('Time (ns)')
        ax.set_ylabel('Raw ADC counts')
        ax.set_title(f'file {self.filename} - {self.tag}, {self.length} runs Beamform')

    def plot_fft(self, ax: plt.Axes=None, f_start=0, f_stop=2000, log = True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        if not self.beamform_wf:
            self.beamform()

        self.beamform_wf.plot_fft(ax=ax, log = log, f_start=f_start, f_stop=f_stop, **kwargs)

    def plot_fft_smoothed(self, ax: plt.Axes=None, f_start=0, f_stop=2000, log = True, window_size=11, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        if not self.beamform_wf:
            self.beamform()

        self.beamform_wf.plot_fft_smoothed(ax=ax, log = log, f_start=f_start, f_stop=f_stop, window_size=window_size, **kwargs)

if __name__ == '__main__':

    current_dir = Path(__file__).resolve()
    parent_dir = current_dir.parents[1]

    basepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_test'
    filename = 'mi1a'

    # basepath = parent_dir / 'data' / 'SURF_Data' / 'beamformertrigger' 
    # filename = '72825_beamformertriggertest1'

    # basepath = parent_dir / 'data' / 'SURF_Data' / '072925_beamformertest1' 
    # filename = '072925_beamformer_6db'

    surf_index = 26
    surf_channel_name = 'AV1'

    info = {'surf_channel_name':surf_channel_name, 'surf_index':surf_index}

    channel = SURFChannelTriggers(basepath=basepath, filename=filename, length=100, info=info)


    fig, ax = plt.subplots()

    channel.plot_beamform_samples(ax=ax)
    
    ax.legend()
    plt.show()