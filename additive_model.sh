#!/bin/bash

# Define the list of items
list_items=("kfr_pooled_pooled_p25"  "kfr_black_pooled_p25"  "jail_black_pooled_p25" "jail_pooled_pooled_p25" "kfr_top20_black_pooled_p25" "kfr_top20_pooled_pooled_p25")

# Use GNU parallel to run all tasks in parallel
# coupled-bootstrap-0.9
parallel -j 6 python covariate_additive_model.py --nsim 100 --est_var {1} '>' logs/cam_output_{1}.log '2>' logs/cam_error_{1}.log ::: "${list_items[@]}"


