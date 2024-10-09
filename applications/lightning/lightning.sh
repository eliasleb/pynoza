#!/bin/sh
for n_points in 35 40 45
do
  n_tail=$((n_points / 10))
  noise_level=0.
  data_case=MTLL
  for order in $(seq 8 2 8)
  do
    for seed in $(seq 0 9)
    do
      python lightning.py --max_order "$order" --n_points $n_points --n_tail $n_tail --seed "$seed" --find_center True \
      --noise_level $noise_level --case $data_case \
      > ../../../git_ignore/lightning_inverse/opt_results/v4_max_order_"${order}"_n_points_"${n_points}"_n_tail_"${n_tail}"_seed_"${seed}"_noise_level_"${noise_level}"_case_$data_case.txt &
    done
  done
done
