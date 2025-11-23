from enum import Enum


class ComponentOptionsBase(Enum):
    def __str__(self):
        return self.value
