list_items=("kfr_pooled_pooled_p25"  "kfr_black_pooled_p25"  "jail_black_pooled_p25" "jail_pooled_pooled_p25" "kfr_top20_black_pooled_p25" "kfr_top20_pooled_pooled_p25")


# 1. Start with NUM_CORES if it exists, otherwise 6
REQUESTED_CORES=${NUM_CORES:-6}

# 2. Cap it at the number of list items
MAX_PARALLEL=${#list_items[@]}                # array length
CORES=$(( REQUESTED_CORES < MAX_PARALLEL ? REQUESTED_CORES : MAX_PARALLEL ))

parallel -j $CORES python empirical_exercise.py --simulator-name weibull --methods indep_gauss,close_npmle,close_gauss,close_gauss_parametric --nsim 100 --est_var {1} '>' logs/weibull_output_{1}.log '2>' logs/weibull_error_{1}.log ::: "${list_items[@]}"
