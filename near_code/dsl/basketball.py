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
    'offense_paint': torch.LongTensor(list(range(42,47))) #binary
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

#not in there: OFFENSE to BASKET!
# class BBallBhBasketDistSelection(AffineFeatureSelectionFunction):
#     def __init__(self, input_size, output_size, num_units):
#         self.full_feature_dim = BBALL_FULL_FEATURE_DIM
#         dist2ball = torch.norm(DEFAULT_BBALL_FEATURE_SUBSETS["offense"]- DEFAULT_BBALL_FEATURE_SUBSETS["ball"], dim=-1)
#         ballhandler_id = torch.argmin(dist2ball).item()
#         ballhandler_pos = DEFAULT_BBALL_FEATURE_SUBSETS["offense"][ballhandler_id]

#         self.feature_tensor = torch.norm(DEFAULT_BBALL_FEATURE_SUBSETS["offense"]- BASKET, dim=-1)
#         super().__init__(input_size, output_size, num_units, name="BhBasketDist")

# class BBallScreenerBasketDistSelection(AffineFeatureSelectionFunction):
#     def __init__(self, input_size, output_size, num_units):
#         self.full_feature_dim = BBALL_FULL_FEATURE_DIM

#         dist2ball = torch.norm(DEFAULT_BBALL_FEATURE_SUBSETS["offense"]- DEFAULT_BBALL_FEATURE_SUBSETS["ball"], dim=-1)
#         ballhandler_id = torch.argmin(dist2ball).item()
#         ballhandler_pos = DEFAULT_BBALL_FEATURE_SUBSETS["offense"][ballhandler_id]
        
#         other_offense = torch.stack([pos for i,pos in enumerate(DEFAULT_BBALL_FEATURE_SUBSETS["offense"]) if i != ballhandler_id])
#         offense2bh = torch.norm(other_offense-ballhandler_pos, dim=-1)

#         screener_id = torch.argmin(offense2bh).item()
#         screener_pos = other_offense[screener_id]
#         self.feature_tensor = torch.norm(screener_pos-BASKET)
#         super().__init__(input_size, output_size, num_units, name="ScreenerBasketDist")


# #easier version: offensedefense distance selection?

# # class BBallInPaint(AffineFeatureSelectionFunction):
# #     def __init__(self, input_size, output_size, num_units):
# #         self.full_feature_dim = BBALL_FULL_FEATURE_DIM

# #         dist2ball = torch.norm(DEFAULT_BBALL_FEATURE_SUBSETS["offense"]- DEFAULT_BBALL_FEATURE_SUBSETS["ball"], dim=-1)
# # # Todo:
# # screener in paint
# # ball in paint
# # def inside_paint(x,y): return (x > 0 and x < 19 and y > 17 and y < 33)

# #     defense2bh = torch.norm(defense-ballhandler_pos, dim=-1)

# # there is a ball-handler # should be discoverable using existing ones? or sohuld this be encoded lol
# # AND there is screener within 10 feet of the ball-handler # uhh this requires fn about screener
# # AND the screener is not in or near the paint #ScreenerPaint
# # AND the screener is no more than 2 ft. further from the basket than the
# #  ball-handler is #ScreenerBasketDistance, BallhandlerBasketDistance
# # AND there is a defensive player <= 12 ft. from the ball-handler (i.e., the
# #  on-ball defender) #Feature for on-ball defender
# # AND the basketball is not in the paint #Should be discoverable?
# #feature of variance?? how to make that atom to atom

# [the ball handler is the same for every frame] <-- feature for who the ball handler is?
# and compute the variance or entropy
# person on offense closest to the ball

