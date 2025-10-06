"""
Pickle files can contain multiple triggers. 
The SURF modules and SURF data are coded around this format particularly well
This python script organises the pickle data
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from SURF_Measurements.surf_data import SURFData

from SURF_Measurements.surf_channel_info import SURFChannelInfo
from SURF_Measurements.surf_unit_info import SURFUnitInfo

from SURF_Measurements.surf_trigger import SURFTrigger
from SURF_Measurements.surf_triggers import SURFTriggers

from SURF_Measurements.surf_channel import SURFChannel
from SURF_Measurements.surf_channel_triggers import SURFChannelTriggers

from SURF_Measurements.surf_unit import SURFUnit
from SURF_Measurements.surf_unit_triggers import SURFUnitTriggers
from SURF_Measurements.surf_units import SURFUnits
from SURF_Measurements.surf_units_triggers import SURFUnitsTriggers

from RF_Utils.Pulse import Pulse
from RF_Utils.Waveform import Waveform

import csv
from pathlib import Path

################
## Getting the data
################

current_dir = Path(__file__).resolve()

parent_dir = current_dir.parents[1]

basepath = parent_dir / 'data' / 'SURF_Data' / 'jjbevents'
filename = 'jjbevents0dB.pkl'
# filename = 'jjbeventsmatched.pkl'
# filename = 'jjbeventsthermal.pkl'

pckl = SURFData(filepath = basepath/filename)
all_data = pckl.format_data_multiple()

# pckl.plot_all(all_data=all_data[0])

################
## Reference pulse for match filtering
################

loaded_list = np.loadtxt(parent_dir / "SURF_Measurements" /"pulse.csv", delimiter=",", dtype=float)
ref_pulse = Pulse(waveform=np.array(loaded_list))

def populate_triggers(info, trigger_type:SURFTrigger = SURFUnit, triggers_type:SURFTriggers = SURFUnitTriggers):
    initial_trigger:SURFTrigger = trigger_type(data=all_data[0], info = info, run=0)

    surf_triggers:SURFTriggers = triggers_type(basepath=basepath, filename=filename, data=[initial_trigger], info = info)
    for run, trigger_data in enumerate(all_data[1:]):
        surf_triggers.add_trigger(run=run+1, surf_data = trigger_data)

    return surf_triggers

def populate_channel(info)->SURFChannelTriggers:
    surf_triggers = populate_triggers(info, trigger_type=SURFChannel, triggers_type=SURFChannelTriggers)
    return surf_triggers


def populate_unit(info)->SURFUnitTriggers:
    surf_triggers = populate_triggers(info, trigger_type=SURFUnit, triggers_type=SURFUnitTriggers)
    return surf_triggers

def populate_units(info)->SURFUnitsTriggers:
    surf_triggers = populate_triggers(info, trigger_type=SURFUnits, triggers_type=SURFUnitsTriggers)
    return surf_triggers


# surf_triggers = populate_unit(info = {"surf_index":26})
# surf_triggers.plot_channel_sum()

surf_triggers = populate_units(info = {"surf_index":[25, 26]})
surf_triggers.plot_channel_sum()
surf_triggers.plot_unit_coherent_sum()
# surf_triggers.plot_channel_matched_sum()

# surf_triggers = populate_channel(info = {"surf_index":26, "rfsoc_channel":7})
# surf_triggers.plot_beamform()
# surf_triggers.plot_coherent_sum()

plt.show()
