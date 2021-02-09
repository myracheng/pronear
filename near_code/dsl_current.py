import dsl


DSL_DICT = {
    ('list', 'list') : [dsl.MapFunction, dsl.MapPrefixesFunction, dsl.SimpleITE],
    ('list', 'atom') : [dsl.FoldFunction, dsl.running_averages.RunningAverageLast5Function, dsl.SimpleITE, 
                        dsl.running_averages.RunningAverageWindow13Function,
                        dsl.running_averages.RunningAverageWindow5Function],
    ('atom', 'atom') : [dsl.AddFunction, dsl.MultiplyFunction, dsl.SimpleITE,
                        dsl.basketball.BBallBallSelection,dsl.basketball.BBallOffenseSelection,
                        dsl.basketball.BBallDefenseSelection,
                        # dsl.basketball.BBallOffenseBallDistSelection,dsl.basketball.BBallOffenseBhDistSelection,dsl.basketball.BBallBhBasketDistSelection,dsl.basketball.BBallScreenerBasketDistSelection,dsl.basketball.BBallBhDefenseDistSelection

                        ]
                        #todo fill in functions!!!
}

# there is a ball-handler # should be discoverable using existing ones? or sohuld this be encoded lol
# AND there is screener within 10 feet of the ball-handler # 
# AND the screener is not in or near the paint
# AND the screener is no more than 2 ft. further from the basket than the
#  ball-handler is
# AND there is a defensive player <= 12 ft. from the ball-handler (i.e., the
#  on-ball defender)
# AND the basketball is not in the paint


# [the ball handler is the same for every frame] <-- feature for who the ball handler is?
# and compute the variance or entropy
# person on offense closest to the ball


CUSTOM_EDGE_COSTS = {
    ('list', 'list') : {},
    ('list', 'atom') : {},
    ('atom', 'atom') : {}
}