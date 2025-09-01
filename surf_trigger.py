import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SURF_Measurements.surf_channel import SURFChannel
from SURF_Measurements.surf_channel_info import SURFChannelInfo
from RF_Utils.Pulse import Pulse

from typing import List, Dict, Any

class SURFTrigger():
    def __init__(self, data:np.ndarray|List[SURFChannel], info:List[SURFChannelInfo]|Dict[str, List], run:int = None, *args, **kwargs):
        
        self.run=run

        self.info:List[SURFChannelInfo]

        if isinstance(info, dict):
            self.info = []
            for values in zip(*info.values()):
                unique_channel_info = dict(zip(info.keys(), values))
                channel_info = SURFChannelInfo(**unique_channel_info)
                self.info.append(channel_info)
        else:
            self.info = info

        if isinstance(data, np.ndarray):
            self.channels = []
            for channel_info in self.info:
                self.channels.append(SURFChannel(data = data[channel_info.surf_index][channel_info.rfsoc_channel], info=channel_info, run=self.run))
        elif isinstance(data[0], SURFChannel):
            self.channels = data

        self.beamform_wf:Pulse = None

    def __len__(self):
        return len(self.channels)
    
    def __iter__(self):
        return iter(self.channels)
    
    def __array__(self):
        return self.channels
    
    def add_channel(self, data, surf_indices:tuple):
        """
        This assumes you have the entire SURF data to add.
        """
        self.channels.append(SURFChannel(data = data[surf_indices[0]][surf_indices[1]], surf_index=surf_indices[0], rfsoc_channel=surf_indices[1], run=self.run))

        # self.channels.sort(key=lambda surf_channel: (surf_channel.info.surf_index, surf_channel.info.rfsoc_channel))

    def remove_channel(self, surf_indices:tuple):
        for i, channel in enumerate(self.channels):
            if (channel.info.surf_index == surf_indices[0] and
                channel.info.rfsoc_channel == surf_indices[1]):
                del self.channels[i]
                break

    def organise_antennae(self):
        """
        Organisation for plotting channels in antenna layout.
        """
        parsed:List[SURFChannel] = []

        for ch in self.channels:
            ant_str = ch.info.antenna.zfill(3)
            row = int(ant_str[0])
            sector = int(ant_str[1:])
            parsed.append((row, sector, ch))

        unique_rows = sorted({r for r, _, _ in parsed})
        unique_sectors = sorted({s for _, s, _ in parsed})

        row_idx_map = {row: i for i, row in enumerate(unique_rows)}
        sector_idx_map = {sector: j for j, sector in enumerate(unique_sectors)}

        return parsed, unique_rows, unique_sectors, row_idx_map, sector_idx_map

    def plot_channels(self, ax: plt.Axes=None):
        """
        This plots all the selected channels in rfsoc order. Each channel is it's own colour (colours loop every 8 channels)
        """
        if ax is None:
            fig, ax = plt.subplots()

        delay = 0

        for channel in self.channels:
            channel.plot_samples(ax=ax, delay = delay)
            delay+=1024

        ax.set_ylim(-1000, 1000)
        ax.legend()

    def plot_antenna_layout(self):
        parsed, unique_rows, unique_sectors, row_idx_map, sector_idx_map = self.organise_antennae()

        fig, axes = plt.subplots(
            nrows=len(unique_rows),
            ncols=len(unique_sectors),
            figsize=(len(unique_sectors)*3, len(unique_rows)*2),
            squeeze=False
        )

        fig.subplots_adjust(hspace=0.4)

        for row, sector, ch in parsed:
            ax = axes[row_idx_map[row]][sector_idx_map[sector]]
            ch.plot_samples(ax=ax)
            ax.set_title(f"{ch.tag}", fontsize=12)
            ax.xaxis.label.set_visible(False)
            # ax.yaxis.label.set_visible(False)
            
        ##Removes empty axis (Antennas of no interrest)
        for i in range(len(unique_rows)):
            for j in range(len(unique_sectors)):
                ax = axes[i][j]
                if len(ax.lines) == 0:
                    ax.axis("off")

    ########################
    ## Beamform stuff
    ########################
    
    def beamform(self, ref_pulse:Pulse = None, window_size = 0.1, min_width=210-15, max_width=210+15, threshold_multiplier=1.8, center_width=5)->Pulse:
        if not ref_pulse:
            current_dir = Path(__file__).resolve()
            parent_dir = current_dir.parents[1]

            loaded_list = np.loadtxt(parent_dir / "SURF_Measurements" /"pulse.csv", delimiter=",", dtype=float)
            ref_pulse = Pulse(waveform=np.array(loaded_list))

        beam = Pulse(waveform=np.zeros(1024), sample_frequency=3e9)

        omitted_channels = []

        for channel in self.channels:
            compare_data = channel.copy()

            max_lag = compare_data.match_filter_check(ref_pulse=ref_pulse, window_size=window_size, min_width=min_width, max_width=max_width, threshold_multiplier=threshold_multiplier, center_width=center_width)

            if not max_lag:
                omitted_channels.append(channel.info.surf_channel_name)
            else:
                beam.waveform += compare_data.waveform

        print(f"Omitted {len(omitted_channels)} Channels\nOmitted Channels : {omitted_channels}")
        self.beamform_wf = beam
        return self.beamform_wf

    def plot_beamform(self, ax: plt.Axes=None, ref_pulse:Pulse = None, window_size = 0.1, min_width=210-15, max_width=210+15, threshold_multiplier=1.8, center_width=5):
        if ax is None:
            fig, ax = plt.subplots()
        if self.beamform_wf is None:
            self.beamform(ref_pulse=ref_pulse, window_size=window_size, min_width=min_width, max_width=max_width, threshold_multiplier=threshold_multiplier, center_width=center_width)
        self.beamform_wf.plot_waveform(ax = ax)
        ax.set_ylabel('Time (ns)')
        ax.set_ylabel('Raw ADC counts')
        ax.set_title('SURF multi-channel beamform')

    def plot_beamform_samples(self, ax: plt.Axes=None, correlation_strength_coef=4, correlation_threshold = 120):
        if ax is None:
            fig, ax = plt.subplots()

        if self.beamform_wf is None:
            self.beamform(correlation_strength_coef, correlation_threshold)
        
        self.beamform_wf.plot_samples(ax = ax)
        ax.set_ylabel('Raw ADC counts')
        ax.set_title(f'SURF {self.info.surf_unit} all channel beamform')

    def plot_beamform_fft(self, ax: plt.Axes=None, f_start=300, f_stop=1200, log = True, scale = 1.0, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        if self.beamform_wf is None:
            self.beamform()

        self.beamform_wf.plot_fft_smoothed(ax=ax, log = log, f_start=f_start, f_stop=f_stop, scale=scale, **kwargs)


if __name__ == '__main__':
    from SURF_Measurements.surf_data import SURFData
    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]
    
    run = 23

    filepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_test' / 'mi1a_150.pkl'

    # filepath = parent_dir / 'data' / 'SURF_Data' / 'beamformertrigger' / '72825_beamformertriggertest1_0.pkl'


    # filepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_all_10dboff' / 'mi1a_35.pkl'

    # filepath = parent_dir / 'data' / 'SURF_Data' / '072925_beamformertest1' / f'072925_beamformer_6db_{run}.pkl'

    surf_data = SURFData(filepath=filepath)
    surf_index = 26


    surf_channel_names = [None]*3
    surf_indies = [26,26,26]
    rfsoc_channels = [1,3,6,7]

    info = {"surf_channel_name":surf_channel_names, "surf_index":surf_indies, "rfsoc_channel":rfsoc_channels}

    surf_channels = SURFTrigger(data = surf_data.format_data(), info=info, run=run)

    # fig, ax = plt.subplots()

    # surf_channels.plot_beamform(ax=ax)

    surf_channels.plot_antenna_layout()


    surf_channels.plot_beamform()

    # surf_channels.remove_channel(surf_indices=(25, 6))

    # surf_channels.plot_antenna_layout()

    plt.show()