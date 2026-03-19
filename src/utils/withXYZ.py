
from dataclasses import dataclass
from senfuslib import NamedArray, AtIndex

@dataclass
class WithXYZ(NamedArray):
    x: AtIndex[0]
    y: AtIndex[1]
    z: AtIndex[2]
    xy: AtIndex[0:2]
