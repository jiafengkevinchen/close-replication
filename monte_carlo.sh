list_items=("kfr_pooled_pooled_p25"  "kfr_white_male_p25"  "kfr_black_male_p25"  "kfr_black_pooled_p25" "kfr_white_pooled_p25" "jail_black_male_p25" "jail_white_male_p25" "jail_black_pooled_p25" "jail_white_pooled_p25" "jail_pooled_pooled_p25" "kfr_top20_black_male_p25" "kfr_top20_white_male_p25" "kfr_top20_black_pooled_p25" "kfr_top20_white_pooled_p25" "kfr_top20_pooled_pooled_p25")

parallel -j 6 python empirical_exercise.py --simulator-name npmle_by_bins --methods all --nsim 1000 --est_var {1} '>' logs/mc_output_{1}.log '2>' logs/mc_error_{1}.log ::: "${list_items[@]}"
