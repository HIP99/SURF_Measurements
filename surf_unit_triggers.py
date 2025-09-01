import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SURF_Measurements.surf_data import SURFData
from SURF_Measurements.surf_channel import SURFChannel
from SURF_Measurements.surf_unit import SURFUnit

from SURF_Measurements.surf_triggers import SURFTriggers

from SURF_Measurements.surf_unit_info import SURFUnitInfo

from RF_Utils.Pulse import Pulse

from typing import List

class SURFUnitTriggers(SURFTriggers):
    """
    Multiple here refers to multiple triggers.

    Requires a basepath to find the files, different triggers (runs), and base file name to append run number
    surf = AV
    surf_index = 0-27
    """
    def __init__(self, info:SURFUnitInfo|dict, basepath:str, filename:str, length:int|tuple=1, data:List[SURFUnit] = None, *args, **kwargs):
        
        self.info:SURFUnitInfo
        
        if isinstance(info, SURFUnitInfo):
            info = SURFUnitInfo(**info.__dict__, **kwargs)
        elif isinstance(info, dict):
            info = SURFUnitInfo(**info)

        super().__init__(basepath=basepath, filename=filename, length=length, data=data, info=info)

        self.tag = f"SURF : {self.info.surf_unit}{self.info.polarisation} / {self.info.surf_index}"

    def populate_triggers(self):
        super().populate_triggers(surf_type = SURFUnit)

    def add_trigger(self, run:int):
        super().add_trigger(surf_type = SURFUnit, run=run)

    def unit_trigger_beamform(self, **kwargs)->List[Pulse]:
        trigger_beamforms:List[Pulse] = []

        for trigger in self.triggers:
            trigger.beamform(**kwargs)
            trigger_beamforms.append(trigger.beamform_wf)

        return trigger_beamforms
    
    def plot_average_beamform(self, ax: plt.Axes=None, correlation_strength_coef=4, correlation_threshold = 120):
        if ax is None:
            fig, ax = plt.subplots()
        if self.beamform_wf is None:
            self.overall_beamform(correlation_strength_coef=correlation_strength_coef, correlation_threshold = correlation_threshold)
        self.beamform_wf.plot_waveform(ax = ax)
        ax.set_ylabel('Time (ns)')
        ax.set_ylabel('Raw ADC counts')
        # ax.set_title(f'Test : {self.basepath.stem} - {self.tag} , {self.length} runs Beamform')
        ax.set_title(f'Test : {self.filename} - {self.tag} , {self.length} runs Beamform')

    def plot_average_beamform_samples(self, ax: plt.Axes=None, correlation_strength_coef=4, correlation_threshold = 120, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        if self.beamform_wf is None:
            self.overall_beamform(correlation_strength_coef=correlation_strength_coef, correlation_threshold = correlation_threshold)
        self.beamform_wf.plot_samples(ax = ax, **kwargs)
        ax.set_ylabel('Raw ADC counts')
        # ax.set_title(f'Test : {self.basepath.stem} - {self.tag} , {self.length} runs Beamform')
        ax.set_title(f'Test : {self.filename} - {self.tag} , {self.length} runs Beamform')

    def plot_channel_beams(self, correlation_strength_coef=4.5, correlation_threshold = 128, **kwargs):
        fig, axs = plt.subplots(4, 2, figsize=(12, 10), sharex=True)

        channel_beams = self.channel_beamform(correlation_strength_coef=correlation_strength_coef, correlation_threshold=correlation_threshold)

        for i in range(8):
            if i < 4:
                channel = channel_beams[i + 4]  # channels 4 to 7
                col = 0
                row = i
            else:
                channel = channel_beams[i - 4]  # channels 0 to 3
                col = 1
                row = i - 4

            channel.plot_samples(ax=axs[row, col])
            axs[row, col].set_title(f"{channel.tag}")
            # stats_text = f"Pulse Quality: {channel.pulse_quality():.2f}"
            # axs[row, col].text(
            #     0.95, 0.95, stats_text,
            #     transform=axs[row, col].transAxes,
            #     fontsize=10,
            #     verticalalignment='top',
            #     horizontalalignment='right',
            #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            # )
            # axs[row, col].set_ylim(-1000, 1000)

        # fig.suptitle(f"Test : {self.basepath.stem} - Channel Beamforms for {self.tag} ({self.length} runs)", fontsize=12)
        fig.suptitle(f"Test : {self.filename} - Channel Beamforms for {self.tag} ({self.length} runs)", fontsize=12)

        plt.tight_layout()

if __name__ == '__main__':

    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    basepath = parent_dir / 'data' / 'SURF_Data' / '072925_beamformertest1' 
    filename = '072925_beamformer_6db'

    basepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_test' 
    filename = 'mi1a'

    # basepath = parent_dir / 'data' / 'SURF_Data' / 'beamformertrigger' 
    # filename = '72825_beamformertriggertest1'

    # basepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_all_10dboff' 
    # filename = 'mi1a'

    # basepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_test2' 
    # filename = 'mi2a'


    surf_index = 26

    info = {'surf_index':surf_index}

    surf_triggers = SURFUnitTriggers(basepath=basepath, filename=filename, length=10, info=info)

    # channels = surf_triggers.channels
    # arr=[]
    # temp = []
    # for channel_arr in channels:
    #     for channel in channel_arr:
    #         temp.append(channel.info.rfsoc_channel)
    #     arr.append(temp)
    #     temp=[]
    # print(arr)

    surf_triggers.plot_channel_beamform()

    plt.show()