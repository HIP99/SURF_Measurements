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

        self.beamform_wf = None


    def __len__(self):
        return len(self.channels[0])
    
    def __iter__(self):
        return iter(self.channels)
    
    def extract_pulse_window(self, pre=20, post=120):
        for channel in self.channels:
            channel.extract_pulse_window(pre=pre, post=post)


    def beamform(self, correlation_strength_coef=4, correlation_threshold = 120):
        sorted_channels = sorted(self.channels, key=lambda x: x.data.hilbert_envelope()[1], reverse=True)
        ref_data = sorted_channels[0].data
        beam = Pulse(waveform=ref_data.waveform.copy(), tag=self.tag + ', New Beamform')
        ref_index = beam.hilbert_envelope()[0]

        omitted_channels = []

        for channel in sorted_channels[1:]:
            compare_data = channel.data.copy()

            pulse_index, pulse_strength = compare_data.hilbert_envelope()

            corr = np.correlate((beam - beam.mean)/beam.std, (compare_data - compare_data.mean)/compare_data.std, mode='full')

            ## Is the found pulses strength is better than cross-correlating
            found_pulse = pulse_strength - np.max(corr) / (np.mean(corr) + correlation_strength_coef * np.std(corr))

            ##If found pulse is good enough
            if found_pulse > 0:
                shift = ref_index - pulse_index
                compare_data.roll(shift=shift)

                ##Improves phase alignment
                best_corr = -np.inf
                best_shift = 0
                for delta in range(-3, 4):
                    test_waveform = np.roll(compare_data.waveform, delta)
                    corr_align = np.correlate(beam.waveform, test_waveform, mode='valid')[0]
                    if corr_align > best_corr:
                        best_corr = corr_align
                        best_shift = delta

                compare_data.roll(shift=best_shift)

                beam.waveform += compare_data

            ##If found pulse isn't good enough
            else:
                ##If correlation isn't great
                if np.max(corr) < correlation_threshold:
                    omitted_channels.append(channel.rfsoc_channel)
                    continue
                lags = np.arange(-len(compare_data) + 1, len(beam))
                max_lag = lags[np.argmax(corr)]

                compare_data.roll(shift=max_lag)
                beam.waveform += compare_data

        print(f"{self.tag} Omitted {len(omitted_channels)} Channels\nOmitted Channels : {omitted_channels}")
        self.beamform_wf = beam

    def old_beamform(self, omit_list:list = []):
        data = Pulse(waveform=self.channels[0].data.waveform, tag=self.tag+', Beamformed')
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

    def plot_beamform(self, ax: plt.Axes=None, correlation_strength_coef=4, correlation_threshold = 120):
        if ax is None:
            fig, ax = plt.subplots()
        if self.beamform_wf is None:
            self.beamform_wf = self.beamform(correlation_strength_coef, correlation_threshold)
        self.beamform_wf.plot_waveform(ax = ax)
        ax.set_ylabel('Time (ns)')
        ax.set_ylabel('Raw ADC counts')
        ax.set_title(f'SURF {self.surf_unit} all channel beamform')

    def plot_beamform_samples(self, ax: plt.Axes=None, correlation_strength_coef=4, correlation_threshold = 120):
        if ax is None:
            fig, ax = plt.subplots()
        if self.beamform_wf is None:
            self.beamform(correlation_strength_coef, correlation_threshold)
        self.beamform_wf.plot_samples(ax = ax)
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
        """Roughly the layout the actual antennas will be"""
        fig, axs = plt.subplots(4, 2, figsize=(12, 10), sharex=True)

        for i in range(8):
            if i < 4:
                channel = self.channels[i + 4]  # channels 4 to 7
                col = 0
                row = i
            else:
                channel = self.channels[i - 4]  # channels 0 to 3
                col = 1
                row = i - 4

            channel.plot_samples(ax=axs[row, col])
            axs[row, col].set_title(f"{channel.data.tag}")
            axs[row, col].set_ylim(-1000, 1000)

        fig.suptitle(f"Channel waveforms for {self.tag}", fontsize=12)

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
    
    run = 23

    filepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_test' / 'mi1a_150.pkl'

    # filepath = parent_dir / 'data' / 'SURF_Data' / 'beamformertrigger' / '72825_beamformertriggertest1_0.pkl'


    # filepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_all_10dboff' / 'mi1a_35.pkl'

    # filepath = parent_dir / 'data' / 'SURF_Data' / '072925_beamformertest1' / f'072925_beamformer_6db_{run}.pkl'

    surf_data = SURFData(filepath=filepath)
    surf_index = 26
    surf_unit = SURFUnit(data = surf_data.format_data(), surf_index=surf_index, run=run)
    surf_unit.plot_unit_grid()

    fig, ax = plt.subplots()

    surf_unit.plot_beamform_samples(ax=ax)

    # fig, ax = plt.subplots()
    # surf_unit.plot_beamform(ax=ax)
    # ax.set_title(f'SURF : 26/AV, all channel beamform, rftrigger_test/mi1a_35.pkl')



    # surf_unit.plot_beamform_fft()

    # surf_unit.extract_pulse_window(pre=25, post=125)
    # surf_unit.plot_fft_grid(scale = len(surf_unit)/2)


    plt.legend()
    plt.show()