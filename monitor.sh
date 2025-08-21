#!/bin/bash

# MONTE CARLO PROGRESS MONITOR
#
# This script monitors the progress of Monte Carlo simulations by counting
# the number of .feather files generated for each outcome variable across
# all simulators simultaneously.
#
# USAGE:
#   # Monitor all simulators (default)
#   ./monitor.sh
#
#   # Monitor with custom refresh rate
#   REFRESH_SECONDS=5 ./monitor.sh
#
#   # One-time check (no live updates)
#   ./monitor.sh --once

# Configuration
REFRESH_SECONDS=${REFRESH_SECONDS:-10}

# Function to get expected files for a simulator
get_expected_files() {
    local simulator="$1"
    case "$simulator" in
        "npmle_by_bins"|"coupled_bootstrap-0.9") echo "1000" ;;
        "weibull") echo "100" ;;
        *) echo "0" ;;
    esac
}

# All outcome variables for npmle_by_bins and coupled_bootstrap-0.9
full_outcome_variables=(
    "kfr_pooled_pooled_p25"
    "kfr_white_male_p25"
    "kfr_black_male_p25"
    "kfr_black_pooled_p25"
    "kfr_white_pooled_p25"
    "jail_black_male_p25"
    "jail_white_male_p25"
    "jail_black_pooled_p25"
    "jail_white_pooled_p25"
    "jail_pooled_pooled_p25"
    "kfr_top20_black_male_p25"
    "kfr_top20_white_male_p25"
    "kfr_top20_black_pooled_p25"
    "kfr_top20_white_pooled_p25"
    "kfr_top20_pooled_pooled_p25"
)

# Subset for weibull (only 6 variables)
weibull_outcome_variables=(
    "kfr_pooled_pooled_p25"
    "kfr_black_pooled_p25"
    "jail_black_pooled_p25"
    "jail_pooled_pooled_p25"
    "kfr_top20_black_pooled_p25"
    "kfr_top20_pooled_pooled_p25"
)

# Function to get outcome variables count for a simulator
get_outcome_count() {
    local simulator="$1"
    if [[ "$simulator" == "weibull" ]]; then
        echo "${#weibull_outcome_variables[@]}"
    else
        echo "${#full_outcome_variables[@]}"
    fi
}

# Function to get outcome variables for a simulator (as array elements)
get_outcome_variables() {
    local simulator="$1"
    if [[ "$simulator" == "weibull" ]]; then
        for var in "${weibull_outcome_variables[@]}"; do
            echo "$var"
        done
    else
        for var in "${full_outcome_variables[@]}"; do
            echo "$var"
        done
    fi
}

# Function to show progress for one simulator
show_simulator_progress() {
    local simulator="$1"
    local expected_per_var=$(get_expected_files "$simulator")
    local data_dir="data/simulated_posterior_means/$simulator"
    
    if [[ ! -d "$data_dir" ]]; then
        echo "  Directory not found: $data_dir"
        return
    fi
    
    local total_completed=0
    local num_vars=$(get_outcome_count "$simulator")
    local total_expected=$((num_vars * expected_per_var))
    
    echo "  Variables: $num_vars, Expected per variable: $expected_per_var"
    
    # Count total files first
    total_completed=$(find "$data_dir" -name "*.feather" 2>/dev/null | wc -l)
    
    # Show individual progress only for incomplete variables
    local incomplete_shown=0
    if [[ "$simulator" == "weibull" ]]; then
        local vars=("${weibull_outcome_variables[@]}")
    else  
        local vars=("${full_outcome_variables[@]}")
    fi
    
    for var in "${vars[@]}"; do
        local var_dir="$data_dir/$var"
        local completed=0
        
        if [[ -d "$var_dir" ]]; then
            completed=$(find "$var_dir" -name "*.feather" 2>/dev/null | wc -l)
        fi
        
        # Only show individual variable progress if not complete and we haven't shown too many
        if [[ $completed -lt $expected_per_var && $incomplete_shown -lt 10 ]]; then
            local percentage=$((completed * 100 / expected_per_var))
            printf "    %-35s %4d/%4d (%3d%%)\n" "$var" "$completed" "$expected_per_var" "$percentage"
            incomplete_shown=$((incomplete_shown + 1))
        fi
    done
    
    if [[ $incomplete_shown -eq 10 ]]; then
        echo "    ... (showing first 10 incomplete variables)"
    fi
    
    local overall_percentage=$((total_completed * 100 / total_expected))
    echo "  TOTAL: $total_completed / $total_expected files ($overall_percentage%)"
    
    if [[ $total_completed -eq $total_expected ]]; then
        echo "  COMPLETED!"
    fi
}

# Function to show current progress for all simulators
show_progress() {
    echo "=== Monte Carlo Progress Monitor ==="
    echo "$(date)"
    echo ""
    
    # Check what simulators actually exist
    local available_simulators=()
    if [[ -d "data/simulated_posterior_means" ]]; then
        for simulator in "npmle_by_bins" "coupled_bootstrap-0.9" "weibull"; do
            if [[ -d "data/simulated_posterior_means/$simulator" ]]; then
                available_simulators+=("$simulator")
            fi
        done
    fi
    
    if [[ ${#available_simulators[@]} -eq 0 ]]; then
        echo "No simulator directories found in data/simulated_posterior_means/"
        echo ""
        echo "Expected simulators:"
        for simulator in "npmle_by_bins" "coupled_bootstrap-0.9" "weibull"; do
            echo "  - $simulator"
        done
        return
    fi
    
    # Show progress for each available simulator
    for simulator in "${available_simulators[@]}"; do
        echo "$simulator:"
        show_simulator_progress "$simulator"
        echo ""
    done
    
    # Overall summary
    echo "=== SUMMARY ==="
    for simulator in "${available_simulators[@]}"; do
        local data_dir="data/simulated_posterior_means/$simulator"
        local total_files=$(find "$data_dir" -name "*.feather" 2>/dev/null | wc -l)
        local num_vars=$(get_outcome_count "$simulator")
        local expected_total=$((num_vars * $(get_expected_files "$simulator")))
        local percentage=$((total_files * 100 / expected_total))
        
        printf "%-25s: %5d / %5d files (%3d%%)\n" "$simulator" "$total_files" "$expected_total" "$percentage"
    done
}

# Handle --once flag for single check
if [[ "$1" == "--once" ]]; then
    show_progress
    exit 0
fi

# Live monitoring
echo "Starting live progress monitor for all simulators..."
echo "Press Ctrl+C to stop"
echo ""

# Use watch if available, otherwise manual loop
if command -v watch >/dev/null 2>&1; then
    # Create a temporary script for watch to execute
    temp_script=$(mktemp)
    
    # Export all functions and variables to the temp script
    {
        declare -f show_progress show_simulator_progress get_outcome_variables get_expected_files get_outcome_count
        declare -p full_outcome_variables weibull_outcome_variables
        echo 'show_progress'
    } > "$temp_script"
    
    watch -n "$REFRESH_SECONDS" "bash $temp_script"
    rm -f "$temp_script"
else
    # Manual refresh loop if watch is not available
    while true; do
        clear
        show_progress
        sleep "$REFRESH_SECONDS"
    done
fi