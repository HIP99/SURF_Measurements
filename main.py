import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from SURF_Measurements.surf_data import SURFData

from SURF_Measurements.surf_unit_info import SURFUnitInfo
from SURF_Measurements.surf_channel_info import SURFChannelInfo

from SURF_Measurements.surf_channel import SURFChannel
from SURF_Measurements.surf_trigger import SURFTrigger

from SURF_Measurements.surf_unit import SURFUnit
from SURF_Measurements.surf_units import SURFUnits

from SURF_Measurements.surf_channel_triggers import SURFChannelTriggers
from SURF_Measurements.surf_triggers import SURFTriggers
from SURF_Measurements.surf_unit_triggers import SURFUnitTriggers
from SURF_Measurements.surf_units_triggers import SURFUnitsTriggers

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
rfsoc_channel = 2
rfsoc_channels = [4, 2]

info_single = {'surf_index':surf_index, 'rfsoc_channel':rfsoc_channel}
info_multiple = {'surf_index':surf_indices, 'rfsoc_channel':rfsoc_channels}

info_unit = {'surf_index':surf_index}
info_unis = {'surf_index':surf_indices}

"""
Do not run without commenting things out. You don't need SURF different SURF instances
"""

#####################
## Surf Channels
#####################

##Single instance
surf_data = SURFData(filepath=filepath)
surf_channel = SURFChannel(data = surf_data.format_data()[surf_index][rfsoc_channel], info=info_single, run = run)

surf_channels = SURFTrigger(data = surf_data.format_data(), info=info_multiple, run=run)

##Multiple triggers
surf_channel_triggers = SURFChannelTriggers(basepath=basepath, filename=filename, length=10, info=info_single)

surf_channels_triggers = SURFTriggers(basepath=basepath, filename=filename, length=10, info=info_multiple)

#####################
## Surf Units
#####################

##Single instance
surf_data = SURFData(filepath=filepath)
surf_unit = SURFUnit(data = surf_data.format_data(), info=info_unit, run = run)
surf_units = SURFUnits(data = surf_data.format_data(), info=info_unis, run = run)

##Multiple triggers
surf_unit_triggers = SURFUnitTriggers(basepath=basepath, filename=filename, length=10, info=info_unit)

##Multiple SURFs, Single instance
surf_units_triggers = SURFUnitsTriggers(basepath=basepath, filename=filename, length=10, info=info_unis)