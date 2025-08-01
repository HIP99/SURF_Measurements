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

class SURFChannelMultiple(SURFChannelInfo):
    """
    Multiple here refers to multiple triggers, not multiple channels.

    Requires a basepath to find the files, different triggers (runs), and base file name to append run number
    surf_channel_name = AV1
    surf_index = 0-27
    channel_index = 0-7, is the rfsoc channel not the surf channel
    """
    def __init__(self, basepath:str, filename:str, length:int=1, data:List[SURFChannel] = None, surf_channel_name:str = None, surf_index:int = None, channel_index:int = None, sample_frequency:int = 3e9, surf_info: dict[str, Any] = {}, channel_info: dict[str, Any] = {}, *args, **kwargs):

        super().__init__(info = channel_info, surf_info=surf_info, surf_channel_name = surf_channel_name, surf_channel_index=(surf_index, channel_index))

        self.basepath = basepath
        self.filename = filename
        self.length = length

        self.triggers = []

        if data is not None:
            self.triggers = data
            self.length = len(self.triggers)

        else:
            """This is just so VSCode will recognise that data points are SURF_Unit instances"""
            filepath = self.basepath / f"{self.filename}_0.pkl"
            surf_data = SURFData(filepath=filepath)
            self.triggers = [SURFChannel(data=surf_data.format_data()[self.surf_index][self.rfsoc_channel], info=self.info, surf_channel_name = self.surf_channel_name, surf_index=self.surf_index, channel_index=self.rfsoc_channel, sample_frequency=sample_frequency, run=0)]
            for run in range(1, length):
                filepath = self.basepath / f"{self.filename}_{run}.pkl"
                surf_data = SURFData(filepath=filepath)
                self.triggers.append(SURFChannel(data=surf_data.format_data()[self.surf_index][self.rfsoc_channel], info=self.info, surf_channel_name = self.surf_channel_name, surf_index=self.surf_index, channel_index=self.rfsoc_channel, sample_frequency=sample_frequency, run=run))
            
            del surf_data

        self.tag = f"SURF : {self.surf_channel_name} / {self.surf_index}.{self.rfsoc_channel}"
        self.beam_wf = None


    def __iter__(self):
        return iter(self.triggers)
    
    def __getitem__(self, run):
        return self.triggers[run]
    
    def __len__(self):
        return len(self.triggers)


    def create_beamform_animation(self, save_path="beamform_animation.mp4"):
        from matplotlib.animation import FuncAnimation
        sorted_triggers = sorted(self.triggers, key=lambda x: x.data.hilbert_envelope(range = (460, 540))[1], reverse=True)
        ref_data = sorted_triggers[0].data
        beam = Pulse(waveform=ref_data.waveform.copy(), tag=self.tag + ', New Beamform')
        ref_index = beam.hilbert_envelope()[0]

        frames = []  # List of (beam_before, trigger, aligned_trigger, beam_after)
        count = 1  # Start with reference trigger
        omitted_triggers = []

        for trigger in sorted_triggers[1:]:
            compare_data = trigger.data.copy()
            pulse_index, pulse_strength = compare_data.hilbert_envelope()

            beam_z = (beam - beam.mean) / beam.std
            comp_z = (compare_data - compare_data.mean) / compare_data.std
            corr = np.correlate(beam_z, comp_z, mode='full')

            found_pulse = pulse_strength - np.max(corr) / (np.mean(corr) + 4 * np.std(corr))
            beam_before = beam.waveform.copy()

            frames.append(beam_before, None, None)

            frames.append(beam_before, compare_data.waveform, None)

            if found_pulse > 0:
                shift = ref_index - pulse_index
                compare_data.roll(shift)

                best_corr = -np.inf
                best_shift = 0
                for delta in range(-3, 4):
                    test_waveform = np.roll(compare_data.waveform, delta)
                    corr_align = np.correlate(beam.waveform, test_waveform, mode='valid')[0]
                    if corr_align > best_corr:
                        best_corr = corr_align
                        best_shift = delta
                compare_data.roll(shift=best_shift)
                aligned = compare_data.waveform.copy()

                beam.waveform += compare_data.waveform

            else:
                if np.max(corr) < 120:
                    omitted_triggers.append(trigger.run)
                    continue
                lags = np.arange(-len(compare_data.waveform) + 1, len(beam.waveform))
                max_lag = lags[np.argmax(corr)]
                compare_data.roll(shift=max_lag)
                aligned = compare_data.waveform.copy()

                beam.waveform += compare_data.waveform
            frames.append((beam_before, aligned, None))

            # frames.append((beam_after))

            count += 1

        # Normalize final beam
        normalized_beam = beam.waveform / count

        frames.append((None, None, None, normalized_beam))

        # Create animation
        fig, ax = plt.subplots(figsize=(10, 5))

        line_beam, = ax.plot([], [], label="Beam", color="blue")
        line_trigger, = ax.plot([], [], label="Trigger", color="red", alpha=0.5)
        line_aligned, = ax.plot([], [], label="Aligned", color="green", alpha=0.5)

        def init():
            ax.set_xlim(0, len(beam.waveform))
            ax.set_ylim(-1.5*np.max(np.abs(beam.waveform)), 1.5*np.max(np.abs(beam.waveform)))
            ax.set_title("Beamforming Process")
            ax.legend()
            return line_beam, line_trigger, line_aligned

        def update(frame):
            beam_before, trigger, aligned, beam_after = frame
            ax.set_title("Beamforming Step")
            if beam_before is not None:
                line_beam.set_data(np.arange(len(beam_before)), beam_before)
                line_trigger.set_data(np.arange(len(trigger)), trigger)
                line_aligned.set_data(np.arange(len(aligned)), aligned)
            else:
                # Final frame
                line_beam.set_data(np.arange(len(beam_after)), beam_after)
                line_trigger.set_data([], [])
                line_aligned.set_data([], [])
                ax.set_title("Final Normalized Beam")
            return line_beam, line_trigger, line_aligned

        anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=300)

        anim.save("beamform_animation.gif", writer='pillow', fps=2)
        plt.close()
        print(f"Animation saved to {save_path}")
    
    def other_beamform(self):
        """
        This is now redundant but is still interresting
        """
        self.triggers = sorted(self.triggers, key=lambda x: x.data.detect_energy()[1], reverse=True)

        beam = Pulse(waveform=self.triggers[0].data.waveform, tag=self.tag+', New Beamform')
        ref_index = beam.hilbert_envelope()[0]

        for i, trigger in enumerate(self.triggers):
            if i == 0:
                continue
            compare_data = trigger.data.copy()

            corr = np.correlate(beam - beam.mean, compare_data - compare_data.mean, mode='full')

            lags = np.arange(-len(compare_data) + 1, len(beam))
            max_lag = lags[np.argmax(corr)]

            compare_data.roll(shift=max_lag)
            beam.waveform += compare_data

        self.beam_wf = beam


    def beamform_attempt(self, correlation_strength_coef=4, correlation_threshold = 120):
        sorted_triggers = sorted(self.triggers, key=lambda x: x.data.hilbert_envelope()[1], reverse=True)

        self.beam_wf = self.beamform_loop(data_list=sorted_triggers, tag=self.tag, correlation_strength_coef=correlation_strength_coef, correlation_threshold=correlation_threshold)

    def beamform(self, reference:Pulse, **kwargs):
        sorted_triggers = sorted(self.triggers, key=lambda x: x.data.hilbert_envelope(range = (460, 540))[1], reverse=True)
        omitted_triggers = []
        
        beam = Pulse(waveform=np.zeros(1024), tag=self.tag + ', New Beamform')
        
        for trigger in sorted_triggers:
            compare_data = trigger.data.copy()
            
            trigger_run = compare_data.match_filter_pulse(reference=reference)
            if trigger_run is not None:
                beam.waveform += compare_data
                
            else:
                omitted_triggers.append(trigger.run)
            
        print(f"Percent of triggers used : {100*(self.length-len(omitted_triggers))/self.length:.2f}%")
        
        self.beam_wf = beam
        
        self.beam_wf.waveform = self.beam_wf.waveform/(self.length-len(omitted_triggers))
        
    def beamform2(self, correlation_strength_coef=4, correlation_threshold = 120):
        """
        Sort the triggers/channels by best pulse
        Set the 'best' pulse as a reference

        Loop through triggers in new order

        If the detected pulse is stronger relative to the correlation with then use pulse location to align with the reference
        Since this may be out of phase/off by +-3 find the best correlation within this window

        If detected pulse isn't good enough/doesn't exist do cross correlation
        If the correlation isn't good enough leave run
        """
        sorted_triggers = sorted(self.triggers, key=lambda x: x.data.hilbert_envelope(range = (460, 540))[1], reverse=True)
        ref_data = sorted_triggers[0].data
        beam = Pulse(waveform=ref_data.waveform.copy(), tag=self.tag + ', Beamform')
        ref_index = beam.hilbert_envelope()[0]

        omitted_triggers = []

        test_arr = []

        i=0
        for trigger in sorted_triggers[1:]:
            i+=1
            compare_data = trigger.data.copy()

            pulse_index, pulse_strength = compare_data.hilbert_envelope()

            corr = np.correlate((beam - beam.mean)/beam.std, (compare_data - compare_data.mean)/compare_data.std, mode='full')

            ## Is the found pulses strength is better than cross-correlating
            found_pulse = pulse_strength - np.max(corr) / (np.mean(corr) + correlation_strength_coef * np.std(corr))

            # if 100 < i < 120:
            #     fig,ax=plt.subplots()
            #     beam.plot_samples(ax=ax)
            

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

            # if 100 < i < 120:
            #     compare_data.plot_samples(ax=ax)
            test_arr.append(beam.pulse_quality())
        fig,ax=plt.subplots()
        ax.plot(test_arr)

        print(f"Triggers {self.basepath} - {self.tag} Omitted {len(omitted_triggers)} triggers\nOmitted triggers : {omitted_triggers}")
        self.beam_wf = beam

    def old_beamform(self):
        """
        This is the old beamform. This works perfectly well for large snr
        """
        beam = Waveform(waveform=self.triggers[0].data.waveform, tag=self.tag+', Old Beamform')
        for trigger in self.triggers[1:]:
            compare_data = trigger.data.copy()

            corr = np.correlate(beam - beam.mean, compare_data - compare_data.mean, mode='full')
            lags = np.arange(-len(compare_data) + 1, len(beam))
            max_lag = lags[np.argmax(corr)]

            compare_data.correlation_align(beam, max_lag)
            beam.waveform += compare_data

        self.beam_wf = beam
    
    def plot_beamform(self, ax: plt.Axes=None, correlation_strength_coef=4, correlation_threshold = 120, mask = slice(None), **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        
        if not self.beam_wf:
            self.beamform(correlation_strength_coef, correlation_threshold)
        self.beam_wf.plot_waveform(ax = ax, mask=mask)
        ax.set_ylabel('Time (ns)')
        ax.set_ylabel('Raw ADC counts')
        ax.set_title(f'{self.tag}, {self.length} runs Beamform')

    def plot_beamform_samples(self, ax: plt.Axes=None, correlation_strength_coef=4, correlation_threshold = 120, mask = slice(None), **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        if not self.beam_wf:
            self.beamform(correlation_strength_coef, correlation_threshold)
        
        self.beam_wf.plot_samples(ax = ax, mask = mask, **kwargs)
        ax.set_ylabel('Time (ns)')
        ax.set_ylabel('Raw ADC counts')
        ax.set_title(f'file {self.filename} - {self.tag}, {self.length} runs Beamform')

    def plot_fft(self, ax: plt.Axes=None, f_start=0, f_stop=2000, log = True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        if not self.beam_wf:
            self.beamform()

        self.beam_wf.plot_fft(ax=ax, log = log, f_start=f_start, f_stop=f_stop, **kwargs)

    def plot_fft_smoothed(self, ax: plt.Axes=None, f_start=0, f_stop=2000, log = True, window_size=11, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        if not self.beam_wf:
            self.beamform()

        self.beam_wf.plot_fft_smoothed(ax=ax, log = log, f_start=f_start, f_stop=f_stop, window_size=window_size, **kwargs)

def get_correlation_threshold(channel:SURFChannelMultiple, correlation_strength_coef=120):
    arr = np.arange(110,180)
    result = []

    for val in arr:
        channel.beamform(correlation_threshold=val, correlation_strength_coef=correlation_strength_coef)
        result.append(channel.beam_wf.pulse_quality())

    return arr, result

def get_correlation_strength_coef(channel:SURFChannelMultiple, correlation_threshold=4):
    arr = np.linspace(3, 8, 20)
    result = []

    for val in arr:
        channel.beamform(correlation_strength_coef=val, correlation_threshold=correlation_threshold)
        result.append(channel.beam_wf.pulse_quality())

    return arr, result

if __name__ == '__main__':

    current_dir = Path(__file__).resolve()
    parent_dir = current_dir.parents[1]

    basepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_test'
    filename = 'mi1a'

    basepath = parent_dir / 'data' / 'SURF_Data' / 'beamformertrigger' 
    filename = '72825_beamformertriggertest1'

    # basepath = parent_dir / 'data' / 'SURF_Data' / '072925_beamformertest1' 
    # filename = '072925_beamformer_6db'

    surf_index = 26
    surf_channel_name = 'AV1'

    channel = SURFChannelMultiple(basepath=basepath, filename=filename, length=100, surf_channel_name=surf_channel_name)

    # fig, ax = plt.subplots()
    # arr, result = get_correlation_strength_coef(channel=channel)
    # ax.plot(arr, result)

    # fig, ax = plt.subplots()
    # arr, result = get_correlation_threshold(channel=channel)
    # ax.plot(arr, result)

    fig, ax = plt.subplots()
    channel.beamform(correlation_strength_coef=4.5, correlation_threshold=128)
    channel.plot_beamform_samples(ax, mask=(slice(250,1024-250)))
    
    ax.legend()
    plt.show()