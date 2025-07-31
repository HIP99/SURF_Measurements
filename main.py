import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from SURF_Measurements.surf_data import SURFData

from SURF_Measurements.surf_channel import SURFChannel
from SURF_Measurements.surf_channel_multiple import SURFChannelMultiple

from SURF_Measurements.surf_unit import SURFUnit
from SURF_Measurements.surf_unit_multiple import SURFUnitMultiple
from SURF_Measurements.surf_units import SURFUnits
from SURF_Measurements.surf_units_multiple import SURFUnitsMultiple

from RF_Utils.Pulse import Pulse
from RF_Utils.Waveform import Waveform

"""
Setup for running programs
"""

current_dir = Path(__file__).resolve()

"""This is over the top for a single instance but needs to be setup for all instances"""
parent_dir = current_dir.parents[1]
basepath = parent_dir / 'data' / 'SURF_Data' / 'rftrigger_test'
filename = 'mi1a'
run = 0
filepath = basepath / f'{filename}_{run}.pkl'

surf_index = 26
surf_indices = [5, 26]
channel_index = 4

"""
Do not run without commenting things out. You don't need SURF different SURF instances
"""

#####################
## Surf Channels
#####################

##Single instance
surf_data = SURFData(filepath=filepath)
surf_channel = SURFChannel(data = surf_data.format_data()[surf_index][channel_index], surf_index = surf_index, channel_index = channel_index, run = run)

##Multiple triggers
surf_channel = SURFChannelMultiple(basepath=basepath, filename=filename, length=10, surf_index = surf_index, channel_index = channel_index)

#####################
## Surf Units
#####################

##Single instance
surf_data = SURFData(filepath=filepath)
surf_unit = SURFUnit(data = surf_data.format_data(), surf_index = surf_index, run = run)

##Multiple triggers
surf_unit = SURFUnitMultiple(basepath=basepath, filename=filename, length=10, surf_index = surf_index)

##Multiple SURFs, Single instance
surf_units = SURFUnits(data = surf_data.format_data(), surf_indices = surf_indices, run = run)

##Multiple SURFs, Multiple triggers
surf_units = SURFUnitsMultiple(basepath=basepath, filename=filename, length=10, surf_indices = surf_indices)