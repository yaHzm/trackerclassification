from ._base import ComponentOptionsBase


class ModelOptions(ComponentOptionsBase):
    DGCNN = "DGCNN"
    ADGCNN = "AffinityDGCNN"
    SINGLE_DGCNN = "SingleDGCNN"
    SAGEConv = "SAGEConv"
    DEEPSET = "DeepSet"


class TrackerOptions(ComponentOptionsBase):
    V4 = "V4"
