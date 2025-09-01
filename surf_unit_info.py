from pathlib import Path
import pandas as pd

from dataclasses import dataclass, asdict, fields

@dataclass
class SURFUnitInfo:
    surf_unit: str|None = None
    surf_index: int|None = None
    polarisation: str|None = None
    address: int|None = None

    def __str__(self):
        return "\n".join(f"{k.replace('_', ' ').title()}: {v or ''}"
                         for k, v in asdict(self).items())
    
    def __iter__(self):
        for f in fields(self):
            yield (f.name, getattr(self, f.name))

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __post_init__(self):

        if self.surf_unit is not None:
            if not (len(self.surf_unit) == 1):
                self.polarisation = self.surf_unit[2] if self.surf_unit.startswith("LF") else self.surf_unit[1]
                self.surf_unit = self.surf_unit[:1] if self.surf_unit.startswith("LF") else self.surf_unit[0]

        if all(getattr(self, f.name) is not None for f in fields(SURFUnitInfo)):
            return
        
        current_dir = Path(__file__).parent
        filepath = current_dir / 'surf_mapping.csv'
        df = pd.read_csv(filepath)

        if self.surf_unit is not None and self.polarisation is not None:
            row = df[(df['Surf Unit'] == self.surf_unit) & (df['Polarisation'] == self.polarisation)]
        elif self.surf_index:
            row = df[df['Surf Index'] == self.surf_index]
        else:
            raise ValueError("No appropriate surf inputted")

        data = row.iloc[0].to_dict()

        self.surf_unit = data.get('Surf Unit')
        self.surf_index = data.get('Surf Index')
        self.polarisation = data.get('Polarisation')
        self.address = data.get('Address')

if __name__ == '__main__':
    channel = SURFUnitInfo(surf_unit='AH')
    print(channel)