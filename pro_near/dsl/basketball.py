import torch
from .library_functions import AffineFeatureSelectionFunction


DEFAULT_BBALL_FEATURE_SUBSETS = {
    "ball"      : torch.LongTensor([0, 1]),
    "offense"   : torch.arange(2,12, dtype = torch.long),
    "defense"   : torch.arange(12,22, dtype = torch.long),
    'offense_dist2ball': torch.arange(22,27, dtype = torch.long),
    'defense_dist2bh': torch.arange(27,32, dtype = torch.long),
    'offense_dist2bh': torch.arange(32,37, dtype = torch.long),
    'offense_dist2basket': torch.arange(37,42, dtype = torch.long),
    'offense_paint': torch.arange(42,47, dtype = torch.long) #binary
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
