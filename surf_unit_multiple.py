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
from SURF_Measurements.surf_unit_info import SURFUnitInfo

from RF_Utils.Pulse import Pulse

from typing import Any, List

class SURFUnitMultiple(SURFUnitInfo):
    """
    Multiple here refers to multiple triggers.

    Requires a basepath to find the files, different triggers (runs), and base file name to append run number
    surf = AV
    surf_index = 0-27
    """
    def __init__(self, basepath:str, filename:str, length:int=1, data:List[SURFUnit] = None, surf:str = None, surf_index:str = None, sample_frequency:int = 3e9, surf_info: dict[str, Any] = {}, *args, **kwargs):
        super().__init__(info = surf_info, surf_unit = surf, surf_index=surf_index)

        self.basepath = basepath
        self.filename = filename
        self.length = length

        if data is not None:
            self.triggers = data
            self.length = len(self.triggers)

        else:
            """This is just so VSCode will recognise that data points are SURF_Unit instances"""
            filepath = self.basepath / f"{self.filename}_0.pkl"
            surf_data = SURFData(filepath=filepath)
            self.triggers = [SURFUnit(data=surf_data.format_data(), surf_info=self.info, surf = self.surf_unit, surf_index=self.surf_index, sample_frequency=sample_frequency)]

            for run in range(1, length):
                filepath = self.basepath / f"{self.filename}_{run}.pkl"
                surf_data = SURFData(filepath=filepath)
                self.triggers.append(SURFUnit(data=surf_data.format_data(), surf_info=self.info, surf = self.surf_unit, surf_index=self.surf_index,sample_frequency=sample_frequency))

        self.tag = f"SURF : {self.surf_unit}{self.polarisation} / {self.surf_index}"

        self.beamform_wf = None

    def __iter__(self):
        return iter(self.triggers)
    
    def __getitem__(self, run):
        return self.triggers[run]
    
    def __len__(self):
        return len(self.triggers)
    
    def get_all_channels(self):
        """This is for overall beamforming"""
        return [
            channel.data
            for trigger in self.triggers
            for channel in trigger.channels
        ]
    
    def get_all_channel_triggers(self, channel_index):
        return [
            trigger.channels[channel_index]
            for trigger in self.triggers
        ]

    def channel_beamform_single(self, channel_triggers:List[SURFChannel], correlation_strength_coef=4, correlation_threshold = 120):
        """This is basically whats in surf_channel_multiple"""
        
        channel_triggers = sorted(channel_triggers, key=lambda x: x.data.hilbert_envelope()[1], reverse=True)
        ref_data = channel_triggers[0].data
        beam = Pulse(waveform=ref_data.waveform.copy(), tag=ref_data.tag + ', New Beamform')
        ref_index = beam.hilbert_envelope()[0]

        del ref_data

        omitted_triggers = []

        for trigger in channel_triggers[1:]:
            compare_data = trigger.data.copy()

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
                    omitted_triggers.append(trigger.run)
                    continue
                lags = np.arange(-len(compare_data) + 1, len(beam))
                max_lag = lags[np.argmax(corr)]

                compare_data.roll(shift=max_lag)
                beam.waveform += compare_data

        # print(f"Triggers {self.basepath} - {self.tag} Omitted {len(omitted_triggers)} triggers\nOmitted triggers : {omitted_triggers}")
        self.beam_wf = beam
        return beam
    
    def channel_beamform(self, correlation_strength_coef=4.5, correlation_threshold = 128):
        channel_beams = []

        for channel_index in range(8):
            channel_beams.append(self.channel_beamform_single(channel_triggers = self.get_all_channel_triggers(channel_index), correlation_strength_coef=correlation_strength_coef, correlation_threshold=correlation_threshold))

        return channel_beams


    def overall_beamform(self, correlation_strength_coef=4, correlation_threshold = 120):
        channels = self.get_all_channels()

        channels = sorted(channels, key=lambda x: x.hilbert_envelope()[1], reverse=True)

        ref_data = channels[0]
        beam = Pulse(waveform=ref_data.waveform.copy(), tag=self.tag + ', Beamform')
        ref_index = beam.hilbert_envelope()[0]
    
        for channel in channels[1:]:
            compare_data = channel.copy()

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
                    continue
                lags = np.arange(-len(compare_data) + 1, len(beam))
                max_lag = lags[np.argmax(corr)]

                compare_data.roll(shift=max_lag)
                beam.waveform += compare_data

        # beam.waveform /= len(self.triggers)
        self.beamform_wf = beam

    def old_beamform(self, omit_list:list = []):
        """
        Average beamform across all triggers
        """
        average_beam = self.triggers[0].beamform(omit_list=omit_list)

        for trigger in self.triggers[1:]:
            compare_beam = trigger.beamform(omit_list=omit_list)

            corr = np.correlate(average_beam - average_beam.mean, compare_beam - compare_beam.mean, mode='full')
            lags = np.arange(-len(compare_beam) + 1, len(average_beam))
            max_lag = lags[np.argmax(corr)]

            compare_beam.roll(shift=max_lag)

            average_beam.waveform += compare_beam
        # average_beam.waveform /= len(self.triggers)
        return average_beam
    
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
        ax.set_title(f'Test : {self.basepath.stem} - {self.tag} , {self.length} runs Beamform')

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
            # Example stats box (customize as needed)
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

def get_correlation_threshold(unit:SURFUnitMultiple, correlation_strength_coef=120):
    arr = np.arange(110,180)
    result = []

    for val in arr:
        unit.overall_beamform(correlation_threshold=val, correlation_strength_coef=correlation_strength_coef)
        result.append(unit.beamform_wf.pulse_quality())

    return arr, result

def get_correlation_strength_coef(unit:SURFUnitMultiple, correlation_threshold=4):
    arr = np.linspace(4, 10, 20)
    result = []

    for val in arr:
        unit.overall_beamform(correlation_strength_coef=val, correlation_threshold=correlation_threshold)
        result.append(unit.beamform_wf.pulse_quality())

    return arr, result
    
if __name__ == '__main__':

    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    basepath = parent_dir / 'data' / 'SURF_Data' / '072925_beamformertest1' 
    filename = '072925_beamformer_6db'

    basepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_test' 
    filename = 'mi1a'

    basepath = parent_dir / 'data' / 'SURF_Data' / 'beamformertrigger' 
    filename = '72825_beamformertriggertest1'

    basepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_all_10dboff' 
    filename = 'mi1a'

    basepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_test2' 
    filename = 'mi2a'
    surf_index = 26

    surf_triggers = SURFUnitMultiple(basepath=basepath, filename=filename, length=100, surf_index=surf_index)

    correlation_strength_coef = 4.325
    correlation_threshold = 117

    # fig, ax = plt.subplots()
    # arr, result = get_correlation_strength_coef(unit=surf_triggers, correlation_threshold=correlation_threshold)
    # ax.plot(arr, result)

    # fig, ax = plt.subplots()
    # arr, result = get_correlation_threshold(unit=surf_triggers, correlation_strength_coef=correlation_strength_coef)
    # ax.plot(arr, result)

    fig, ax = plt.subplots()

    surf_triggers.plot_average_beamform_samples(ax=ax, correlation_strength_coef=correlation_strength_coef, correlation_threshold=correlation_threshold)

    surf_triggers.plot_channel_beams(correlation_strength_coef=correlation_strength_coef, correlation_threshold=correlation_threshold)

    plt.show()