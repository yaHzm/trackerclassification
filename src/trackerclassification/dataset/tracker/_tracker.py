from __future__ import annotations
from pydantic import BaseModel

from ._code import TrackerCode
from ._pose import TrackerPose
from ._geometry import TrackerGeometry
from ...utils.typing import Matrix_7x3_f


class Tracker(BaseModel):
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
        code (TrackerCode): a 3-digit unique identifier (unique within a sample of trackers)
        pose (TrackerPose):  rigid transformation (R, t) that places the tracker in 3D space 
        geometry (TrackerGeometry): the 3D coordinates of its LEDs in the tracker's local coordinate system
    """
    code: TrackerCode
    pose: TrackerPose
    geometry: TrackerGeometry

    @property
    def id(self) -> int:
        """
        Get the unique id of the tracker (in [0, 26]) based on its code.

        Returns:
            int: The unique id of the tracker
        """
        return self.code.to_id()
    
    def get_leds_world_coords(self) -> Matrix_7x3_f:
        """
        Get the 3D coordinates of the LEDs in world coordinates.

        Returns:
            Matrix_7x3_f: An array of shape (7, 3) containing the 3D coordinates of the LEDs in world coordinates
        """
        R, t = self.pose.R, self.pose.t
        leds_tracker = self.geometry.as_array()
        leds_tracker_centered = leds_tracker - self.geometry.center    # shift so center is at (0,0,0)
        leds_world = (R @ leds_tracker_centered.T).T + t
        return leds_world