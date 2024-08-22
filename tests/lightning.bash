#!/bin/sh
scale=1e6
for n_points in 100
do
  n_tail=10
  for order in $(seq 0 16 2)
  do
    for seed in $(seq 0 9)
    do
      python lightning.py --max_order "$order" --n_points $n_points --n_tail $n_tail --order_scale $scale --seed "$seed" --find_center True > ../../../git_ignore/lightning_inverse/opt_results/max_order_"${order}"_scale_"${scale}"_n_points_"${n_points}"_n_tail_"${n_tail}"_seed_"${seed}".txt &
    done
  done
done
