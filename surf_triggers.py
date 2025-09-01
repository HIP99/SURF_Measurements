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

        self.beamform_wf:Pulse = None
                
    def __iter__(self):
        return iter(self.triggers)
    
    def __getitem__(self, run):
        return self.triggers[run]
    
    def __array__(self):
        return self.triggers
    
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

    def add_trigger(self, surf_type:SURFTrigger, run:int):
        surf_data = self.get_trigger_data(run=run)
        self.triggers.append(surf_type(data=surf_data.format_data(), info=self.info))

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

    def beamform_channels(self, **kwargs) -> SURFTrigger:
        channel_beamform:List[SURFChannel] = []
        channel_trigger_arr = self.channels_triggers
        for channel_trigger in channel_trigger_arr:
            channel_trigger.beamform(**kwargs)
            channel_beamform.append(SURFChannel(data=channel_trigger.beamform_wf, info=channel_trigger.info, tag = channel_trigger.tag))

        channel_beamform_instance = SURFTrigger(data=channel_beamform, info=[ch.info for ch in channel_beamform])
        return channel_beamform_instance

    def beamform_triggers(self, **kwargs) -> List[Pulse]:
        trigger_beamform_arr:List[Pulse] = []
        for trigger in self.triggers:
            trigger.beamform(**kwargs)
            trigger_beamform_arr.append(trigger.beamform_wf)

        return trigger_beamform_arr

    def beamform(self, ref_pulse:Pulse = None, window_size = 0.1, min_width=210-15, max_width=210+15, threshold_multiplier=1.8, center_width=5) -> Pulse:
        if not ref_pulse:
            current_dir = Path(__file__).resolve()
            parent_dir = current_dir.parents[1]

            loaded_list = np.loadtxt(parent_dir / "SURF_Measurements" /"pulse.csv", delimiter=",", dtype=float)
            ref_pulse = Pulse(waveform=np.array(loaded_list))

        beam = Pulse(waveform=np.zeros(1024), sample_frequency=3e9)

        omitted_events = []

        for trigger in self.triggers:
            for channel in trigger.channels:
                compare_data = channel.copy()

                max_lag = compare_data.match_filter_check(ref_pulse=ref_pulse, window_size=window_size, min_width=min_width, max_width=max_width, threshold_multiplier=threshold_multiplier, center_width=center_width)

                if not max_lag:
                    omitted_events.append((channel.info.surf_channel_name, channel.run))
                else:
                    beam.waveform += compare_data.waveform

        print(f"Omitted {len(omitted_events)} Events\nOmitted Events : {omitted_events}")
        self.beamform_wf = beam
        return self.beamform_wf

    ########
    ## Plots
    ########

    def plot_channel(self, trigger:int, channel_index:tuple, ax: plt.Axes=None, **kwargs):
        channel = self.get_channel_trigger(trigger=trigger, channel_index=channel_index)

        channel.plot_samples(ax=ax, **kwargs)
        ax.set_ylabel("Raw ADC counts")

    def plot_beamform(self, ax: plt.Axes=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        beamform:Pulse = self.beamform()

        beamform.plot_samples(ax = ax, **kwargs)
        ax.set_ylabel('Raw ADC counts')
        ax.set_title(f'Test : {self.filename}, {self.length} runs Beamform')

    def plot_trigger_beamform(self, ax: plt.Axes=None, **kwargs):
        """
        Due to the number of triggers this may be horrible
        """
        if ax is None:
            fig, ax = plt.subplots()

    def plot_channel_beamform(self):
        channel_beamforms:SURFTrigger = self.beamform_channels()
        channel_beamforms.plot_antenna_layout()



if __name__ == '__main__':
    from SURF_Measurements.surf_data import SURFData
    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    basepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_test' 
    filename = 'mi1a'


    surf_index = 26
    surf_channel_name = ['AV1', 'AV3', 'AV6', 'AV8']

    info = {'surf_channel_name':surf_channel_name}

    surf_triggers = SURFTriggers(basepath=basepath, filename=filename, length=5, info=info)

    surf_triggers.plot_channel_beamform()

    plt.show()