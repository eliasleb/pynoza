#!/bin/sh
n_points=60
n_tail=6
noise_level=0.
for order in $(seq 0 2 8)
do
  for seed in $(seq 0 9)
  do
    python lightning.py --max_order "$order" --n_points $n_points --n_tail $n_tail --seed "$seed" --find_center True \
    --noise_level $noise_level \
    > ../../git_ignore/lightning_inverse/opt_results/max_order_"${order}"_n_points_"${n_points}"_n_tail_"${n_tail}"_seed_"${seed}"_noise_level_"${noise_level}".txt &
  done
done
