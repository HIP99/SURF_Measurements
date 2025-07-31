import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd

from RF_Utils.MIE_Channel import MIE_Channel
from SURF_Measurements.surf_unit_info import SURFUnitInfo
from pathlib import Path

from typing import Any

class SURFChannelInfo(SURFUnitInfo, MIE_Channel):
    """
    Handles the SURF Channel and MIE Channel infomation

    This inherits from SURFUnitInfo since it uses all the information SURFUnitInfo gets

    Currently doesn't handle LF channels very well

    Inputted info is assumed to be correct with no checks.
    """
    def __init__(self, info: dict[str, Any] = {}, surf_info: dict[str, Any] = {}, surf_channel_name:str = None, surf_channel_index:tuple = None, *args, **kwargs):

        self.info = info

        if not self.info:
            if surf_channel_name is not None:
                self._parse_surf_name(surf=surf_channel_name)
            elif surf_channel_index is not None:
                self._parse_surf_index(surf=surf_channel_index)

            if surf_channel_name is None and surf_channel_index is None:
                ValueError("You have not enterred a valid SURF channel")

            self.get_surf_channel_info(surf_info = surf_info, surf_channel_name=surf_channel_name, surf_channel_index=surf_channel_index)

    @property
    def rfsoc_channel(self):
        return int(self.info["RFSoC Channel"])    

    def _parse_surf_name(self, surf: str):
        if len(surf) != 3 and len(surf) != 4:
            raise ValueError(f"Invalid SURF format: '{surf}'. Expected, RX box cable - Polarisation - SURF Channel.")

        if surf.startswith("LF"):
            unit, pol, channel = surf[:1], surf[2], surf[3]
        else:
            unit, pol, channel = surf[0], surf[1], surf[2]
            if unit not in "ABCDEFGIJKLM":
                raise ValueError(f"Invalid RX box cable letter: '{unit}'")
            
        if pol not in "HV":
            raise ValueError(f"Invalid polarisation: '{pol}'")
        
        if not channel.isdigit() or not (1 <= int(channel) <= 8):
            raise ValueError(f"Invalid SURF Channel number: '{channel}'")
        
    def _parse_surf_index(self, surf: str):
        if len(surf) != 2:
            raise ValueError(f"Invalid SURF format: '{surf}'. Expected, (SURF Data Index, RFSoC Channel)")

        index, channel = surf[0], surf[1]

        if not (0 <= int(index) <= 27):
            raise ValueError(f"Invalid SURF Channel number: '{index}'")
        if not (0 <= int(channel) <= 7):
            raise ValueError(f"Invalid SURF Channel number: '{channel}'")

    def get_surf_channel_info(self, surf_info: dict[str, Any] = {}, surf_channel_name:str = None, surf_channel_index:tuple = None):

        surf_channel = None
        rfsoc_channel = None

        surf_unit = None
        surf_index = None
        surf_pol = None

        ##Gets individual bits of info required for the csv's
        if surf_channel_name is not None:
            surf_unit = surf_channel_name[:1] if surf_channel_name.startswith("LF") else surf_channel_name[0]
            surf_channel = surf_channel_name[-1]
            surf_pol = surf_channel_name[2] if surf_channel_name.startswith("LF") else surf_channel_name[1]
        if surf_channel_index is not None:
            surf_index = surf_channel_index[0]
            rfsoc_channel = surf_channel_index[-1]

        ##SURF mapping
        if not surf_info:
            if surf_unit:
                self.get_surf_info(surf_unit=surf_unit+surf_pol)
            elif surf_index:
                self.get_surf_info(surf_index=surf_index)
        else:
            self.info = surf_info

        ##Channel mapping first
        current_dir = Path(__file__).parent
        filepath = current_dir / 'surf_channel_mapping.csv'
        df = pd.read_csv(filepath)

        if surf_channel is not None and rfsoc_channel is None:
            row = df[df['SURF Channel'] == int(surf_channel)]
            rfsoc_channel = row['RFSoC Channel'].iloc[0]

        elif rfsoc_channel is not None and surf_channel is None:
            row = df[df['RFSoC Channel'] == int(rfsoc_channel)]
            surf_channel = row['SURF Channel'].iloc[0]

        self.info.update({"RFSoC Channel" : rfsoc_channel})

        if surf_channel_name is not None:
            self.get_info(surf=surf_channel_name)
        else:
            self.get_info(surf=f"{self.info['Surf Unit']}{self.info['Polarisation']}{surf_channel}")

        # print(self.info)

if __name__ == '__main__':

    channel = SURFChannelInfo(surf_channel_name='AV2')
    # channel = SURFChannelInfo(surf_channel_index=(15,4))

    print(channel.info)
    print(channel)