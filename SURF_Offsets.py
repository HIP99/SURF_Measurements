import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from SURF_Data import SURF_Data
from SURF_Offset import SURF_Offset
from MIE_Channel import MIE_Channel

class SURF_Offsets(SURF_Data):
    def __init__(self, clock_cycle=10, *args, **kwargs):
        self.clock_cycle = clock_cycle

        self.surf_list = []
        self.channel_list = []

        self.current_dir = Path(__file__).parent
        self.filepath = self.current_dir / 'data' / 'Offset_Data'

        self.offsets = []

    def get_offsets(self, length = 100, factor:int = None, height = 50):
        temp = []
        for surf in self.surf_list:
            for channel in self.channel_list:
                for i in range(length):
                    surf_run = SURF_Offset(filepath= self.filepath / f'{surf}_{channel}_{i}.pkl')
                    if factor:
                        surf_run.data.upsampleFreqDomain(factor)
                    temp.append(surf_run.data.find_pulse_peak(height=height))
                self.offsets.append(temp)
                temp=[]

    def offset_stats(self):
        std = np.std(self.offsets)
        mean = np.mean(self.offsets)
        return std, mean

    def plot_histogram(self, ax: plt.Axes=None, bins=20):
        ax.hist(self.offsets, bins=bins, edgecolor='black')
        ax.set_title(f'Histogram : offsets for {self.clock_cycle}')
        ax.set_xlabel('Offset Value')
        ax.set_ylabel('Frequency')

        std, mean = self.offset_stats()

        stats_text = f"Std Dev. : {std:.2f}\nMean : {mean:.2f} Sample"
        
        ax.text(0.97, 0.97, stats_text, verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))
        
    def plot_stacked_histogram(self, ax: plt.Axes=None, bins=20):
        ax.hist(self.offsets, bins=bins, stacked=True, edgecolor='black')
        ax.set_title(f'Histogram : offsets for {self.clock_cycle}')
        ax.set_xlabel('Offset Value')
        ax.set_ylabel('Frequency')

        std, mean = self.offset_stats()

        stats_text = f"Std Dev. : {std:.2f}\nMean : {mean:.2f} Sample"
        
        ax.text(0.97, 0.97, stats_text, verticalalignment='top', horizontalalignment='right',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

if __name__ == '__main__':
    surf_num = 5, 
    channel_num = 6
    clock_cycle = 10
    factor = None

    surf = SURF_Offsets(surf_num = surf_num, channel_num=channel_num, clock_cycle=clock_cycle)
    surf.get_offsets(length = 100, factor = factor)

    fig, ax = plt.subplots()

    surf.plot_histogram(ax=ax, bins = 10)

    plt.show()


