import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from RF_Utils.MIE_Channel import MIE_Channel
from SURF_Measurements.surf_unit_info import SURFUnitInfo

from dataclasses import dataclass, asdict, fields

@dataclass
class SURFChannelInfo(SURFUnitInfo, MIE_Channel):
    rfsoc_channel: int|None = None
    surf_channel: int|None = None
    
    # def __iter__(self):
    #     for f in fields(self):
    #         yield (f.name, getattr(self, f.name))

    # def to_dict(self):
    #     return {f.name: getattr(self, f.name) for f in fields(self)}

    def __post_init__(self):
        if all(value is not None for value in asdict(self).values()):
            return
        
        if self.surf_channel_name is not None:
            self.surf_channel = int(self.surf_channel_name[3]) if self.surf_channel_name.startswith("LF") else int(self.surf_channel_name[2])
            self.polarisation = self.surf_channel_name[2] if self.surf_channel_name.startswith("LF") else self.surf_channel_name[1]
            self.surf_unit = self.surf_channel_name[:1] if self.surf_channel_name.startswith("LF") else self.surf_channel_name[0]

        if all(getattr(self, f.name) is not None for f in fields(SURFUnitInfo)):
            pass
        else:
            SURFUnitInfo.__post_init__(self=self)

        if self.surf_channel is not None and self.rfsoc_channel is None:
            self.rfsoc_channel = (int(self.surf_channel)+3)%8

        elif self.rfsoc_channel is not None and self.surf_channel is None:
            self.surf_channel = 1 + (int(self.rfsoc_channel) + 4) % 8

        self.surf_channel_name = self.surf_unit + self.polarisation + str(self.surf_channel)

        MIE_Channel.__post_init__(self=self)

if __name__ == '__main__':

    unit = SURFUnitInfo(surf_unit='AH')

    info = {'rfsoc_channel' : 3}

    # channel = SURFChannelInfo(**unit.__dict__, surf_channel=3)
    # channel = SURFChannelInfo(**info)
    # channel = SURFChannelInfo(**unit.__dict__, **info)
    channel = SURFChannelInfo(**unit.__dict__, rfsoc_channel = 3)

    print(dict(channel))
    print(channel.to_dict())
    print(channel.__dict__)

    print(channel.rfsoc_channel)