from __future__ import annotations 
from typing_extensions import override
from pydantic import model_validator

import logging
LOGGER = logging.getLogger(__name__)

from ._base import TrackerCodeBase


class V4TrackerCode(TrackerCodeBase):
    """
    A 3-digit code (c0, c1, c2), with each digit in {0, 1, 2}, specifiying the arrangement of the LEDs of a v4 tracker.
    There are 27 possible codes and codes are unique within a sample of trackers. Those are the possible codes:

    000 / 001 / 002 / 010 / 011 / 012 / 020 / 021 / 022   
    100 / 101 / 102 / 110 / 111 / 112 / 120 / 121 / 122   
    200 / 201 / 202 / 210 / 211 / 212 / 220 / 221 / 222   

    The code encodes the arrangement as follows:

                   ^            
                  / \               
               0 /   \ 2             
                /     \            
               /       \            
              /         \           
      side 2 / 1       1 \  side 1 
            /             \           
           /               \         
          /                 \        
       2 /                   \ 0       
        /          1          \    
        -----------------------      
             0   side 0  2   

    - c0 describes the arrangement of the LEDs on side 0 (indices 3 and 4).   
       In order to be able to get rid of rotational symmetry, side 0 is always coded "inversely" compared to the other two sides, 
       which means that two LEDs are used on this side and the non-existent third defines the code for side 0.
    - c1 describes the arrangement of the LEDs on side 1 (index 5). The existing LED defines the code for side 1.
    - c2 describes the arrangement of the LEDs on side 2 (index 6). The existing LED defines the code for side 2.

    Example: 
    code = (2, 0, 1) means that side 0 has LEDs 0 and 1 (and no LED 2), side 1 has LED 0, and side 2 has LED 1.

    Each code can be mapped to a unique id in [0, 26] and vice versa. The equations are:   
    id = c0 * 9 + c1 * 3 + c2    

    c0 = id // 9   
    c1 = (id // 3) % 3   
    c2 = id % 3   

    Attributes:
        c0 (int): code digit for side 0 (in {0, 1, 2})
        c1 (int): code digit for side 1 (in {0, 1, 2})
        c2 (int): code digit for side 2 (in {0, 1, 2})
    """
    c0: int
    c1: int
    c2: int

    @model_validator(mode="after")
    def check_range(self):
        """
        Validation to ensure that c0, c1, c2 are in {0, 1, 2}.
    
        Parameters:
            values (dict): Dictionary of field values

        Returns:
            dict: The original values if validation passes

        Raises:
            ValueError: If any of c0, c1, or c2 are not in {0, 1, 2}
        """
        for name, val in (("c0", self.c0), ("c1", self.c1), ("c2", self.c2)):
            if not (0 <= val <= 2):
                LOGGER.error("%s must be between 0 and 2 (got %d)", name, val)
                raise ValueError(f"{name} must be between 0 and 2")
        return self
    
    @override
    @classmethod
    def num_unique_ids(cls) -> int:
        return 27

    def as_tuple(self) -> tuple[int, int, int]:
        """
        Convert the V4TrackerCode to a tuple representation

        Returns:
            tuple[int,int,int]: A tuple (c0, c1, c2) representing the V4TrackerCode
        """
        return (self.c0, self.c1, self.c2)

    def __getitem__(self, idx: int) -> int:
        """
        Allow indexing to access individual code digits
        Parameters:
            idx (int): Index of the code digit to access (0, 1, or 2)

        Returns:
            int: The code digit at the specified index

        Raises:
            IndexError: If idx is not in {0, 1, 2}
        """
        if idx == 0:
            return self.c0
        elif idx == 1:
            return self.c1
        elif idx == 2:
            return self.c2
        else:
            raise IndexError("V4TrackerCode index must be 0, 1, or 2")
        
    @override
    @classmethod
    def from_id(cls, id: int) -> V4TrackerCode:
        if not (0 <= id < cls.num_unique_ids()):
            LOGGER.error("id must be between 0 and 26 (got %d)", id)
            raise ValueError("id must be between 0 and 26")
        c0 = id // 9
        c1 = (id // 3) % 3
        c2 = id % 3
        return cls(c0=c0, c1=c1, c2=c2)
    
    @override
    def to_id(self) -> int:
        return self.c0 * 9 + self.c1 * 3 + self.c2