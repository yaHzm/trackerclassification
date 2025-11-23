from __future__ import annotations

from ._code import V4TrackerCode
from ._geometry import V4TrackerGeometry
from ._base import TrackerBase


class V4Tracker(TrackerBase):
    """
    Representation of a version 4 tracker. 

    A tracker can be visualized as following:

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

    Trackers are equilateral triangles with 7 LEDs in total:
        - 3 LEDs at the corners (indices 0, 1, 2)
        - 2 LEDs on side 0 (indices 3, 4)
        - 1 LED on side 1 (index 5)
        - 1 LED on side 2 (index 6)

    A tracker is specified by the following attribues:
    Attributes:
        code (V4TrackerCode): a 3-digit unique identifier (unique within a sample of trackers)
        pose (TrackerPose):  rigid transformation (R, t) that places the tracker in 3D space 
        geometry (TrackerGeometry): the 3D coordinates of its LEDs in the tracker's local coordinate system
    """
    CodeClass = V4TrackerCode
    GeometryClass = V4TrackerGeometry