import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from SURF_Measurements.SURF_Data import SURF_Data
from SURF_Measurements.SURF_Channel import SURF_Channel
from RF_Utils.Pulse import Pulse
from RF_Utils.Waveform import Waveform

class SURF_Unit(SURF_Data):
    def __init__(self, filepath:str, surf:str = None, surf_index:int = None, run:int=None, *args, **kwargs):

        self.surf = surf
        self.surf_index = surf_index
        self.run = run
        self.data = []

        super().__init__(filepath = filepath, *args, **kwargs)

    def get_surf_index(self):
        self.surf_index = self.surf_mapping.index(self.surf)

    def format_data(self):
        all_data = super().format_data()

        if self.surf:
            self.get_surf_index()

        for i in range(8):
            self.data.append(SURF_Channel(filepath=None, data=all_data[self.surf_index][i], surf = self.surf, surf_index = self.surf_index, channel_index = i, run=self.run))


    def plot_unit_series(self, ax: plt.Axes=None):
        if ax is None:
            fig, ax = plt.subplots()

        delay = 0
        for channel in self.data:
            channel.plot_samples(ax=ax, delay = delay)
            delay+=1024
        ax.legend()

    def plot_unit_grid(self):
        fig, axs = plt.subplots(4, 2, figsize=(12, 10), sharex=True)

        for i, channel in enumerate(self.data):
            row = i // 2
            col = i % 2
            channel.plot_samples(ax=axs[row, col])
            axs[row, col].set_title(f"{channel.data.tag}")
            axs[row, col].set_ylim(-1000, 1000)

        plt.tight_layout()

if __name__ == '__main__':
    surf = "IH"
    channel = 8
    run = 0
    surf_index = 5
    channel_index = 2

    current_dir = Path(__file__).resolve()

    parent_dir = current_dir.parents[1]

    filepath = parent_dir / 'data' / 'SURF_Data' / f'SURF{surf}{channel}' / f'SURF{surf}{channel}_{run}.pkl'

    unit = SURF_Unit(filepath = filepath, surf=surf, surf_index = surf_index, run=None)
    unit.format_data()

    fig, ax = plt.subplots()
    unit.plot_unit_series(ax=ax)

    unit.plot_unit_grid()

    # plt.legend()
    plt.show()