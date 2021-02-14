import torch
from .library_functions import AffineFeatureSelectionFunction


DEFAULT_BBALL_FEATURE_SUBSETS = {
    "ball"      : torch.LongTensor([0, 1]),
    "offense"   : torch.LongTensor(list(range(2,12))),
    "defense"   : torch.LongTensor(list(range(12,22))),
    'offense_dist2ball': torch.LongTensor(list(range(22,27))),
    'defense_dist2bh': torch.LongTensor(list(range(27,32))),
    'offense_dist2bh': torch.LongTensor(list(range(32,37))),
    'offense_dist2basket': torch.LongTensor(list(range(37,42))),
    'offense_paint': torch.LongTensor(list(range(42,47))), #binary
}

BBALL_FULL_FEATURE_DIM = 47


class BBallBallSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = DEFAULT_BBALL_FEATURE_SUBSETS["ball"]
        super().__init__(input_size, output_size, num_units, name="BallXYAffine")

class BBallOffenseSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = DEFAULT_BBALL_FEATURE_SUBSETS["offense"]
        super().__init__(input_size, output_size, num_units, name="OffenseXYAffine")

class BBallDefenseSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = DEFAULT_BBALL_FEATURE_SUBSETS["defense"]
        super().__init__(input_size, output_size, num_units, name="DefenseXYAffine")
    
class BBallOffenseBallDistSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = DEFAULT_BBALL_FEATURE_SUBSETS["offense_dist2ball"]
        super().__init__(input_size, output_size, num_units, name="OffenseBallDist")

class BBallDefenseBhDistSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = DEFAULT_BBALL_FEATURE_SUBSETS["defense_dist2bh"]
        super().__init__(input_size, output_size, num_units, name="DefenseBhDist")

class BBallOffenseBhDistSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = DEFAULT_BBALL_FEATURE_SUBSETS["offense_dist2bh"]
        super().__init__(input_size, output_size, num_units, name="OffenseBhDist")

class BBallOffenseBasketDistSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = DEFAULT_BBALL_FEATURE_SUBSETS["offense_dist2basket"]
        super().__init__(input_size, output_size, num_units, name="OffenseBasketDist")

class BBallOffensePaintSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = DEFAULT_BBALL_FEATURE_SUBSETS["offense_paint"]
        super().__init__(input_size, output_size, num_units, name="OffenseInPaint")

class BBallBallPaintSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = DEFAULT_BBALL_FEATURE_SUBSETS["ball_inpaint"]
        super().__init__(input_size, output_size, num_units, name="BallInPaint")

class BBallScreenPaintSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = DEFAULT_BBALL_FEATURE_SUBSETS["screen_inpaint"]
        super().__init__(input_size, output_size, num_units, name="ScreenInPaint")

class BBallScreenBhDistSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = DEFAULT_BBALL_FEATURE_SUBSETS["screen_bh_dist"]
        super().__init__(input_size, output_size, num_units, name="ScreenBhDist")

class BBallBhOneHotSelection(AffineFeatureSelectionFunction):
    def __init__(self, input_size, output_size, num_units):
        self.full_feature_dim = BBALL_FULL_FEATURE_DIM
        self.feature_tensor = DEFAULT_BBALL_FEATURE_SUBSETS["ballhandler"]
        super().__init__(input_size, output_size, num_units, name="BallhandlerId")