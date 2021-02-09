
python3.8 train.py \
--algorithm rnn \
--exp_name ballscreen \
--trial 1 \
--train_data data/helpers/allskip5/train_fullfeatures.npy \
--valid_data data/helpers/allskip5/test_fullfeatures.npy \
--test_data data/helpers/allskip5/test_fullfeatures.npy \
--train_labels data/helpers/allskip5/train_ballscreens.npy \
--valid_labels data/helpers/allskip5/test_ballscreens.npy \
--test_labels data/helpers/allskip5/test_ballscreens.npy \
--input_type "list" \
--output_type "list" \
--input_size 47 \
--output_size 2 \
--num_labels 1 \
--lossfxn "crossentropy" \
--max_depth 8 \
--frontier_capacity 8 \
--learning_rate 0.001 \
--neural_epochs 10 \
--symbolic_epochs 15 \
--max_num_units 16 \
--min_num_units 4 \
--class_weights "0.1,0.9"


python3.8 train.py \
--algorithm astar-near \
--exp_name ballscreen \
--trial 1 \
--train_data data/helpers/allskip5/train_fullfeatures.npy \
--valid_data data/helpers/allskip5/test_fullfeatures.npy \
--test_data data/helpers/allskip5/test_fullfeatures.npy \
--train_labels data/helpers/allskip5/train_ballscreens.npy \
--valid_labels data/helpers/allskip5/test_ballscreens.npy \
--test_labels data/helpers/allskip5/test_ballscreens.npy \
--input_type "list" \
--output_type "list" \
--input_size 47 \
--output_size 2 \
--num_labels 1 \
--lossfxn "crossentropy" \
--max_depth 8 \
--frontier_capacity 8 \
--learning_rate 0.001 \
--neural_epochs 6 \
--symbolic_epochs 15 \
--max_num_units 16 \
--min_num_units 4 \
--class_weights "0.1,0.9"

#ballscreen w/ og features only
python3.8 train.py \
--algorithm astar-near \
--exp_name ballscreen \
--trial 1 \
--train_data data/helpers/allskip5/train_raw_trajs.npy \
--valid_data data/helpers/allskip5/test_raw_trajs.npy \
--test_data data/helpers/allskip5/test_raw_trajs.npy \
--train_labels data/helpers/allskip5/train_ballscreens.npy \
--valid_labels data/helpers/allskip5/test_ballscreens.npy \
--test_labels data/helpers/allskip5/test_ballscreens.npy \
--input_type "list" \
--output_type "list" \
--input_size 22 \
--output_size 2 \
--num_labels 1 \
--lossfxn "crossentropy" \
--max_depth 8 \
--frontier_capacity 8 \
--learning_rate 0.001 \
--neural_epochs 6 \
--symbolic_epochs 15 \
--max_num_units 16 \
--min_num_units 4 \
--class_weights "0.1,0.9"

# ballhandler
python3.8 train.py \
--algorithm astar-near \
--exp_name ballhandler \
--trial 1 \
--train_data data/helpers/allskip5/train_raw_trajs.npy \
--valid_data data/helpers/allskip5/test_raw_trajs.npy \
--test_data data/helpers/allskip5/test_raw_trajs.npy \
--train_labels data/helpers/allskip5/train_ballhandlers.npy \
--valid_labels data/helpers/allskip5/test_ballhandlers.npy \
--test_labels data/helpers/allskip5/test_ballhandlers.npy \
--input_type "list" \
--output_type "list" \
--input_size 22 \
--output_size 6 \
--num_labels 6 \
--lossfxn "crossentropy" \
--max_depth 8 \
--frontier_capacity 8 \
--learning_rate 0.001 \
--neural_epochs 6 \
--symbolic_epochs 15 \
--penalty 0.01 \
--max_num_units 16 \
--min_num_units 4
