#!/bin/bash

# Define the list of items
list_items=("kfr_pooled_pooled_p25"  "kfr_white_male_p25"  "kfr_black_male_p25"  "kfr_black_pooled_p25" "kfr_white_pooled_p25" "jail_black_male_p25" "jail_white_male_p25" "jail_black_pooled_p25" "jail_white_pooled_p25" "jail_pooled_pooled_p25" "kfr_top20_black_male_p25" "kfr_top20_white_male_p25" "kfr_top20_black_pooled_p25" "kfr_top20_white_pooled_p25" "kfr_top20_pooled_pooled_p25")

# Use GNU parallel to run all tasks in parallel
# coupled-bootstrap-0.9
parallel -j 5 python empirical_exercise.py --simulator-name coupled_bootstrap-0.9 --methods all --nsim 1000 --est_var {1} '>' logs/output_{1}.log '2>' logs/error_{1}.log ::: "${list_items[@]}"

