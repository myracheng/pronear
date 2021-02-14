import dsl


DSL_DICT = {
    ('list', 'list') : [dsl.MapFunction, dsl.MapPrefixesFunction, dsl.SimpleITE],
    ('list', 'atom') : [dsl.FoldFunction, dsl.running_averages.RunningAverageLast5Function, dsl.SimpleITE, 
                        dsl.running_averages.RunningAverageWindow13Function,
                        dsl.running_averages.RunningAverageWindow5Function],
    ('atom', 'atom') : [dsl.AddFunction, dsl.MultiplyFunction, dsl.SimpleITE,
                        dsl.basketball_47.BBallBallSelection,dsl.basketball_47.BBallOffenseSelection,
                        dsl.basketball_47.BBallDefenseSelection,
                        dsl.basketball_47.BBallOffenseBallDistSelection,dsl.basketball_47.BBallOffenseBhDistSelection,
                        dsl.basketball_47.BBallOffenseBasketDistSelection,dsl.basketball_47.BBallDefenseBhDistSelection,
                        dsl.basketball_47.BBallOffensePaintSelection
                        ]
}


CUSTOM_EDGE_COSTS = {
    ('list', 'list') : {},
    ('list', 'atom') : {},
    ('atom', 'atom') : {}
}