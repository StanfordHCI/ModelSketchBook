"""
msb_enums
---
Shared set of enums for the model_sketch_book API
"""

from enum import Enum


class InputType(Enum):
    Text = 1
    Image = 2
    ImageLocal = 3
    Number = 4
    GroundTruth = 5


class OutputType(Enum):
    Binary = 1
    MultiClass = 2
    Continuous = 3


class SketchMode(Enum):
    Train = 1
    Test = 2


class idMode(Enum):
    Dataset = 1
    Concept = 2
    Sketch = 3


class ModelType(Enum):
    LinearRegression = 1
    LogisticRegression = 2
    MLP = 3
    DecisionTree = 4
    RandomForest = 5
    ZeroShot = 6
    ManualLinear = 7


class SketchSortMode(Enum):
    SketchPred = 1
    GroundTruth = 2
    Diff = 3
    AbsDiff = 4
