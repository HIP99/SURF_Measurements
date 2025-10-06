import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SURF_Measurements.surf_data import SURFData
from SURF_Measurements.surf_channel import SURFChannel
from SURF_Measurements.surf_channel_triggers import SURFChannelTriggers
from SURF_Measurements.surf_trigger import SURFTrigger
from SURF_Measurements.surf_unit import SURFUnit
from SURF_Measurements.surf_units import SURFUnits

from SURF_Measurements.surf_unit_info import SURFUnitInfo
from SURF_Measurements.surf_channel_info import SURFChannelInfo

from RF_Utils.Pulse import Pulse

from typing import List

class SURFTriggers():
    """
    Base Class for handling multiple triggers

    Multiple triggers assumes that each trigger is _{trigger run}
    """
    def __init__(self, basepath:str, filename:str, length:int|tuple=1, data:List[SURFTrigger] = None, info:List[SURFChannelInfo]|dict[str,List] = None, *args, **kwargs):

        self.basepath = basepath
        self.filename = filename
        self.length = length

        self.triggers : List[SURFTrigger]
        self.info : List[SURFChannelInfo]

        self.tag=None

        if isinstance(info, dict):
            self.info = []
            for values in zip(*info.values()):
                unique_channel_info = dict(zip(info.keys(), values))
                channel_info = SURFChannelInfo(**unique_channel_info)
                self.info.append(channel_info)

        elif isinstance(info, SURFUnitInfo):
            self.info = info

        elif isinstance(info[0], (SURFChannelInfo, SURFUnitInfo)):
            self.info = info

        if data:
            if isinstance(data, list):
                if isinstance(data[0], SURFTrigger):
                    self.triggers = data
                    self.length = len(self.triggers)
                else:
                    raise TypeError("Data inputs must be a list of SURF Trigger instances")
        else:
            self.triggers = []
            self.populate_triggers()
                
    def __iter__(self):
        return iter(self.triggers)
    
    def __getitem__(self, run)->SURFTrigger:
        return self.triggers[run]
    
    def __array__(self)->list[SURFTrigger]:
        return self.triggers
    
    def __len__(self):
        return len(self.triggers) * len(self.triggers[0])
    
    ########
    ## Data Handling
    ########
    
    def get_trigger_data(self, run):
        filepath = self.basepath / f"{self.filename}_{run}.pkl"
        surf_data = SURFData(filepath=filepath)
        return surf_data
    
    def populate_triggers(self, surf_type:SURFTrigger=SURFTrigger):
        for run in range(self.length):
            surf_data = self.get_trigger_data(run=run)
            self.triggers.append(surf_type(data=surf_data.format_data(), info=self.info, run=run))

    def add_trigger(self, surf_type:SURFTrigger, run:int, surf_data=None):
        if surf_data is None:
            surf_data = self.get_trigger_data(run=run)
            surf_data = surf_data.format_data()

        self.triggers.append(surf_type(data=surf_data, info=self.info, run=run))

    @property
    def channels(self) -> List[List[SURFChannel]]:
        return list(map(list, zip(*self.triggers)))

    @property
    def channels_triggers(self) -> List[SURFChannelTriggers]:
        return [
            SURFChannelTriggers(data=list(channel_trigger), info=channel_trigger[0].info)
            for channel_trigger in zip(*self.triggers)
        ]
    
    def get_channel_trigger(self, trigger:int, channel_index:tuple) -> SURFChannel:
        for channel in self.triggers[trigger]:
            if channel_index == (channel.info.surf_index, channel.info.rfsoc_channel):
                return channel
            
        print(f"SURF index : {channel_index[0]}, RFSoC Channel : {channel_index[1]} NOT FOUND in trigger : {trigger}")

    def get_channel_triggers(self, channel_index:tuple) -> List[SURFChannel]:
        """
        Returns an array for a single channel for all triggers
        """
        channel_arr:List[SURFChannel]

        for channel in self.channels:
            if channel_index == (channel[0].info.surf_index, channel[0].info.rfsoc_channel):
                channel_arr = channel
                return channel_arr
        
        print(f"SURF index : {channel_index[0]}, RFSoC Channel : {channel_index[1]} NOT FOUND")
        return None
    
    ########
    ## Beamforming
    ########

    def coherent_sum_channels(self) -> Pulse:
        channel_sum:List[SURFChannel] = []
        channel_trigger_arr = self.channels_triggers
        for channel_trigger in channel_trigger_arr:
            channel_sum.append(SURFChannel(data=channel_trigger.coherent_sum(), info=channel_trigger.info, tag = channel_trigger.tag))

        channel_sum_instance = SURFTrigger(data=channel_sum, info=[ch.info for ch in channel_sum])
        return channel_sum_instance

    def coherent_sum_triggers(self) -> Pulse:
        trigger_sum_arr:List[Pulse] = []
        for trigger in self.triggers:
            trigger_sum_arr.append(trigger.coherent_sum())

        return trigger_sum_arr

    def coherent_sum(self) -> Pulse:
        all_channel_trigger_arr = [ch for trigger in self.triggers for ch in trigger]
        sorted_data = sorted(all_channel_trigger_arr, key=lambda surf_channel: surf_channel.snr, reverse=True)
        summed_waveform = Pulse(waveform=sorted_data[0].copy().waveform, tag = f"{self.tag} Sum", sample_frequency=3e9)
        
        ## alignment drift can be avoided by centering the sum near the beginning
        check_index = 10

        for channel in sorted_data[1:check_index]:
            compare_data = channel.copy()
            compare_data.cross_correlate(ref_pulse=summed_waveform)
            summed_waveform.waveform += compare_data.waveform

        summed_waveform.center_on_peak()

        for channel in sorted_data[check_index:]:
            compare_data = channel.copy()
            compare_data.cross_correlate(ref_pulse=summed_waveform)
            summed_waveform.waveform += compare_data.waveform

        return summed_waveform

    def matched_sum_channels(self, **kwargs) -> SURFTrigger:
        channel_matched_sum:List[SURFChannel] = []
        channel_trigger_arr = self.channels_triggers
        for channel_trigger in channel_trigger_arr:
            channel_matched_sum.append(SURFChannel(data=channel_trigger.matched_sum(**kwargs), info=channel_trigger.info, tag = channel_trigger.tag))

        channel_matched_sum_instance = SURFTrigger(data=channel_matched_sum, info=[ch.info for ch in channel_matched_sum])
        return channel_matched_sum_instance

    def matched_sum_triggers(self, **kwargs) -> List[Pulse]:
        trigger_matched_sum_arr:List[Pulse] = []
        for trigger in self.triggers:
            trigger.matched_sum(**kwargs)
            trigger_matched_sum_arr.append(trigger.matched_sum(**kwargs))

        return trigger_matched_sum_arr

    def matched_sum(self, ref_pulse:Pulse = None, window_size = 0.1, min_width=210-15, max_width=210+15, threshold_multiplier=1.8, center_width=5) -> Pulse:
        if not ref_pulse:
            current_dir = Path(__file__).resolve()
            parent_dir = current_dir.parents[1]

            loaded_list = np.loadtxt(parent_dir / "SURF_Measurements" /"pulse.csv", delimiter=",", dtype=float)
            ref_pulse = Pulse(waveform=np.array(loaded_list), tag = "ref_pulse")

        matched_sum = Pulse(waveform=np.zeros(1024), sample_frequency=3e9, tag=f"{self.tag} matched sum")

        omitted_events = []

        for trigger in self.triggers:
            for channel in trigger.channels:
                compare_data = channel.copy()

                max_lag = compare_data.match_filter_check(ref_pulse=ref_pulse, window_size=window_size, min_width=min_width, max_width=max_width, threshold_multiplier=threshold_multiplier, center_width=center_width)

                if not max_lag:
                    omitted_events.append((channel.info.surf_channel_name, channel.run))
                else:
                    matched_sum.waveform += compare_data.waveform

        percent_omitted = 100 * len(omitted_events) / len(self)
        print(f"Omitted {len(omitted_events)} Events - {percent_omitted:.4g}%")#\nOmitted Events : {omitted_events}")

        return matched_sum

    ########
    ## Plots
    ########

    def plot_channel(self, trigger:int, channel_index:tuple, ax: plt.Axes=None, **kwargs):
        channel = self.get_channel_trigger(trigger=trigger, channel_index=channel_index)

        channel.plot_samples(ax=ax, **kwargs)
        ax.set_ylabel("Raw ADC counts")

    def plot_matched_sum(self, ax: plt.Axes=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        matched_sum = self.matched_sum()
        matched_sum.plot_samples(ax = ax, **kwargs)
        ax.set_ylabel('Raw ADC counts')
        ax.set_title(f'Test : {self.filename}, {self.length} runs matched sum')

    def plot_trigger_matched_sum(self, ax: plt.Axes=None, **kwargs):
        """
        Due to the number of triggers this may be horrible
        """
        if ax is None:
            fig, ax = plt.subplots()

    def plot_channel_matched_sum(self):
        channel_matched_sums:SURFTrigger = self.matched_sum_channels()
        channel_matched_sums.plot_antenna_layout()


    def plot_sum(self, ax: plt.Axes=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        coherent_sum:Pulse = self.coherent_sum()

        coherent_sum.plot_samples(ax = ax, **kwargs)
        ax.set_ylabel('Raw ADC counts')
        ax.set_title(f'Test : {self.filename}, {self.length} runs Coherent Sum')

    def plot_trigger_sum(self, ax: plt.Axes=None, **kwargs):
        """
        Due to the number of triggers this may be horrible
        """
        if ax is None:
            fig, ax = plt.subplots()

    def plot_channel_sum(self):
        channel_sums:SURFTrigger = self.coherent_sum_channels()
        channel_sums.plot_antenna_layout()


if __name__ == '__main__':
    from SURF_Measurements.surf_data import SURFData
    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    basepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_test' 
    filename = 'mi1a'


    surf_index = 26
    surf_channel_name = ['AV1', 'AV3', 'AV6', 'AV8']

    info = {'surf_channel_name':surf_channel_name}

    surf_triggers = SURFTriggers(basepath=basepath, filename=filename, length=500, info=info)

    # surf_triggers.plot_beamform()
    # surf_triggers.plot_sum()
    surf_triggers.plot_channel_matched_sum()
    surf_triggers.plot_channel_sum()
    plt.show()