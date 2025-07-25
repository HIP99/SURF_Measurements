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

if __name__ == '__main__':
    channel = SURFUnitInfo(surf_unit='AH')

    print(channel['Polarisation'])
    print(channel)