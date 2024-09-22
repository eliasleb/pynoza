#!/bin/sh
n_points=100
n_tail=10
for order in $(seq 0 2 16)
do
  for seed in $(seq 0 9)
  do
    python lightning.py --max_order "$order" --n_points $n_points --n_tail $n_tail --seed "$seed" --find_center True \
    > ../../git_ignore/lightning_inverse/opt_results/max_order_"${order}"_n_points_"${n_points}"_n_tail_"${n_tail}"_seed_"${seed}".txt &
done
