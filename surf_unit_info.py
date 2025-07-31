import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pathlib import Path

import pandas as pd

from typing import Any

class SURFUnitInfo():
    """
    Handles the SURF Unit infomation

    Currently doesn't handle LF channels very well

    Inputted info is assumed to be correct with no checks.
    """
    def __init__(self, info: dict[str, Any] = {}, surf_unit:str = None, surf_index:int = None, *args, **kwargs):

        self.info = info

        if not self.info:
            self.get_surf_info(surf_unit=surf_unit, surf_index=surf_index)

    def __str__(self):
        lines = [f"{k}: {v}" for k, v in self.info.items()]
        return "\n".join(lines)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.info})"

    def __getitem__(self, key):
        return self.info[key]
    
    def __contains__(self, key):
        return key in self.info

    def keys(self):
        return self.info.keys()

    def values(self):
        return self.info.values()

    def items(self):
        return self.info.items()

    @property
    def surf_unit(self):
        return self.info["Surf Unit"]
    
    @property
    def surf_index(self):
        return self.info["Surf Index"]    
    
    @property
    def polarisation(self):
        return self.info["Polarisation"]

    @property
    def address(self):
        return self.info["Address"]
    
    def get_surf_info(self, surf_unit:str = None, surf_index:tuple = None):
        """
        Assumes surf_unit is in the form AV or IH etc
        """
        current_dir = Path(__file__).parent
        filepath = current_dir / 'surf_mapping.csv'
        df = pd.read_csv(filepath)

        if surf_unit:
            surf_pol = surf_unit[2] if surf_unit.startswith("LF") else surf_unit[1]
            surf_unit = surf_unit[:1] if surf_unit.startswith("LF") else surf_unit[0]
            row = df[(df['Surf Unit'] == surf_unit) & (df['Polarisation'] == surf_pol)]

        elif surf_index:
            row = df[df['Surf Index'] == int(surf_index)]

        if not row.empty:
            self.info = row.iloc[0].to_dict()

    def beamform_loop(self, data_list, tag, correlation_strength_coef=4, correlation_threshold = 120, **kwargs):
        """
        Data list should be sorted first
        Set the 'best' pulse as a reference

        Loop through triggers in new order

        If the detected pulse is stronger relative to the correlation with then use pulse location to align with the reference
        Since this may be out of phase/off by +-3 find the best correlation within this window

        If detected pulse isn't good enough/doesn't exist do cross correlation
        If the correlation isn't good enough leave run
        """
        from RF_Utils.Pulse import Pulse
        import numpy as np
        ref_data = data_list[0].data
        beam = Pulse(waveform=ref_data.waveform.copy(), tag=tag + ', Beamform')
        ref_index = beam.hilbert_envelope()[0]

        for data in data_list[1:]:

            compare_data = data.data.copy()

            pulse_index, pulse_strength = compare_data.hilbert_envelope()
            corr = np.correlate((beam - beam.mean)/beam.std, (compare_data - compare_data.mean)/compare_data.std, mode='full')
            found_pulse = pulse_strength - np.max(corr) / (np.mean(corr) + correlation_strength_coef * np.std(corr))

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
            return beam


if __name__ == '__main__':
    channel = SURFUnitInfo(surf_unit='AH')

    print(channel['Polarisation'])
    print(channel)