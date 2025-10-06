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

        self.get_info(info, **kwargs)

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

    def __iter__(self):
        return iter(self.triggers)
    
    def __getitem__(self, run):
        return self.triggers[run]
    
    def __len__(self):
        return len(self.triggers)
    
    def get_trigger_data(self, run):
        filepath = self.basepath / f"{self.filename}_{run}.pkl"
        surf_data = SURFData(filepath=filepath)
        return surf_data
    
    def add_trigger(self, run:int, surf_data=None):
        if surf_data is None:
            surf_data = self.get_trigger_data(run=run)
            surf_data = surf_data.format_data()
        
        if surf_data.ndim == 3:
            surf_data = surf_data[self.info.surf_index][self.info.rfsoc_channel]
        elif surf_data.ndim == 2:
            surf_data = surf_data[self.info.rfsoc_channel]

        self.triggers.append(SURFChannel(data=surf_data, info=self.info, run=run))
    
    def get_info(self, info:SURFChannelInfo|dict = None, **kwargs):
        if isinstance(info, SURFChannelInfo):
            self.info = info
        elif isinstance(info, dict):
            self.info = SURFChannelInfo(**info)
        else:
            raise TypeError("Infomation input is not in the correct form.")
    
    def coherent_sum(self) -> Pulse:
        # sorted_triggers = sorted(self.triggers, key=lambda surf_channel: surf_channel.total_energy, reverse=True)
        sorted_triggers = sorted(self.triggers, key=lambda surf_channel: surf_channel.snr, reverse=True)
        summed_waveform = Pulse(waveform=sorted_triggers[0].copy().waveform, tag = self.tag+" Sum", sample_frequency=3e9)

        ## alignment drift can be avoided by centering the sum near the beginning
        check_index = 5

        for channel in sorted_triggers[1:check_index]:
            compare_data = channel.copy()
            compare_data.cross_correlate(ref_pulse=summed_waveform)
            summed_waveform.waveform += compare_data.waveform

        summed_waveform.center_on_peak()

        for channel in sorted_triggers[check_index:]:
            compare_data = channel.copy()
            compare_data.cross_correlate(ref_pulse=summed_waveform)
            summed_waveform.waveform += compare_data.waveform

        return summed_waveform

    def matched_sum(self, ref_pulse:Pulse = None, window_size = 0.1, min_width=210-15, max_width=210+15, threshold_multiplier=1.8, center_width=5)->Pulse:
        if not ref_pulse:
            current_dir = Path(__file__).resolve()
            parent_dir = current_dir.parents[1]

            loaded_list = np.loadtxt(parent_dir / "SURF_Measurements" /"pulse.csv", delimiter=",", dtype=float)
            ref_pulse = Pulse(waveform=np.array(loaded_list))

        matched_sum = Pulse(waveform=np.zeros(1024), sample_frequency=3e9, tag = self.tag+" matched sum")

        omitted_triggers = []

        for channel in self.triggers:

            compare_data = channel.copy()

            max_lag = compare_data.match_filter_check(ref_pulse=ref_pulse, window_size=window_size, min_width=min_width, max_width=max_width, threshold_multiplier=threshold_multiplier, center_width=center_width)

            if not max_lag:
                omitted_triggers.append(channel.run)
            else:
                matched_sum.waveform += compare_data.waveform

        percent_omitted = 100 * len(omitted_triggers) / len(self)
        print(f"{self.tag}\nOmitted {len(omitted_triggers)} Triggers - {percent_omitted:.4g}%\n")#Omitted Triggers : {omitted_triggers}")

        return matched_sum
        
    def plot_matched_sum(self, ax: plt.Axes=None, mask = slice(None), **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        matched_sum = self.matched_sum()
        matched_sum.plot_samples(ax = ax, mask = mask, **kwargs)
        ax.set_ylabel('Time (ns)')
        ax.set_ylabel('Raw ADC counts')
        ax.set_title(f'file {self.filename} - {self.tag}, {self.length} runs matched sum')

    def plot_coherent_sum(self, ax: plt.Axes=None):
        if ax is None:
            fig, ax = plt.subplots()
        coherent_sum=self.coherent_sum()
        coherent_sum.plot_samples(ax = ax)
        ax.set_ylabel('Samples')
        ax.set_ylabel('Raw ADC counts')
        ax.set_title('SURF multi-channel coherent sum')

    def plot_fft(self, ax: plt.Axes=None, f_start=0, f_stop=2000, log = True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        matched_sum = self.matched_sum()
        matched_sum.plot_fft(ax=ax, log = log, f_start=f_start, f_stop=f_stop, **kwargs)

    def plot_fft_smoothed(self, ax: plt.Axes=None, f_start=0, f_stop=2000, log = True, window_size=11, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        matched_sum = self.matched_sum()
        matched_sum.plot_fft_smoothed(ax=ax, log = log, f_start=f_start, f_stop=f_stop, window_size=window_size, **kwargs)

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

    channel.plot_matched_sum(ax=ax)
    channel.plot_coherent_sum(ax=ax)
    
    ax.legend()
    plt.show()