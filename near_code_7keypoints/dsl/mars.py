import torch
from .library_functions import AffineFeatureSelectionFunction

OFFSET = 158
# MARS_FEATURE_SUBSETS = {
#     "angle_head_body" : torch.LongTensor([25, 26, OFFSET + 25, OFFSET + 26]),
#     "axis_ratio" : torch.LongTensor([29, OFFSET + 29]),
#     "speed" : torch.LongTensor([34, 38, OFFSET + 34, OFFSET + 38]),
#     "acceleration" : torch.LongTensor([36, OFFSET + 36]),
#     "resh_twd_itrhb" : torch.LongTensor([39]),
#     "velocity" : torch.LongTensor([62, 63, OFFSET + 62, OFFSET + 63]),
#     "rel_angle" : torch.LongTensor([60, 49, 61, OFFSET + 49, OFFSET + 61]),
#     "rel_dist" : torch.LongTensor([50, 51, 53, 54]),
#     "area_ellipse_ratio" : torch.LongTensor([59])
# }
MARS_FEATURE_SUBSETS = {
    "angle_head_body" : torch.arange(0, 4, dtype = torch.long),
    "axis_ratio" : torch.arange(4, 6, dtype = torch.long),
    "speed" : torch.arange(6, 10, dtype = torch.long),
    "acceleration" : torch.arange(10, 12, dtype = torch.long),
    "resh_twd_itrhb" : torch.arange(12, 13, dtype = torch.long),
    "velocity" : torch.arange(13, 17, dtype = torch.long),
    "rel_angle" : torch.arange(17, 22, dtype = torch.long),
    "rel_dist" : torch.arange(22, 26, dtype = torch.long),
    "area_ellipse_ratio" : torch.arange(26, 27, dtype = torch.long)
}

MARS_INDICES = [25, 26, OFFSET + 25, OFFSET + 26, 29, OFFSET + 29, 34, 38, OFFSET + 34, OFFSET + 38, 36, OFFSET + 36, \
                39, 62, 63, OFFSET + 62, OFFSET + 63, 60, 49, 61, OFFSET + 49, OFFSET + 61, 50, 51, 53, 54, 59]
MARS_FULL_FEATURE_DIM = len(MARS_INDICES)
# MARS_FULL_FEATURE_DIM = 316

class MarsAngleHeadBodySelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MARS_FULL_FEATURE_DIM
        self.feature_tensor = MARS_FEATURE_SUBSETS["angle_head_body"]
        super().__init__(input_size, output_size, num_units, name="AngleHeadBodySelect")

class MarsAxisRatioSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MARS_FULL_FEATURE_DIM
        self.feature_tensor = MARS_FEATURE_SUBSETS["axis_ratio"]
        super().__init__(input_size, output_size, num_units, name="AxisRatioSelect")

class MarsSpeedSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MARS_FULL_FEATURE_DIM
        self.feature_tensor = MARS_FEATURE_SUBSETS["speed"]
        super().__init__(input_size, output_size, num_units, name="SpeedSelect")

class MarsVelocitySelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MARS_FULL_FEATURE_DIM
        self.feature_tensor = MARS_FEATURE_SUBSETS["velocity"]
        super().__init__(input_size, output_size, num_units, name="VelocitySelect")

class MarsAccelerationSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MARS_FULL_FEATURE_DIM
        self.feature_tensor = MARS_FEATURE_SUBSETS["acceleration"]
        super().__init__(input_size, output_size, num_units, name="AccelerationSelect")

class MarsResidentTowardIntruderSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MARS_FULL_FEATURE_DIM
        self.feature_tensor = MARS_FEATURE_SUBSETS["resh_twd_itrhb"]
        super().__init__(input_size, output_size, num_units, name="ResidentTowardIntruderSelect")

class MarsRelAngleSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MARS_FULL_FEATURE_DIM
        self.feature_tensor = MARS_FEATURE_SUBSETS["rel_angle"]
        super().__init__(input_size, output_size, num_units, name="RelativeAngleSelect")

class MarsRelDistSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MARS_FULL_FEATURE_DIM
        self.feature_tensor = MARS_FEATURE_SUBSETS["rel_dist"]
        super().__init__(input_size, output_size, num_units, name="RelativeDistanceSelect")

class MarsAreaEllipseRatioSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = MARS_FULL_FEATURE_DIM
        self.feature_tensor = MARS_FEATURE_SUBSETS["area_ellipse_ratio"]
        super().__init__(input_size, output_size, num_units, name="AreaEllipseRatioSelect")