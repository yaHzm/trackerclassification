from ._base import ComponentOptionsBase


class ModelOptions(ComponentOptionsBase):
    DGCNN = "DGCNN"
    SAGEConv = "SAGEConv"
    DEEPSET = "DeepSet"


class TrackerOptions(ComponentOptionsBase):
    V4 = "V4"
