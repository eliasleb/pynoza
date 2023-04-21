#!/bin/zsh
for order in {7..8}
do
  for n_points in 45
  do
    for seed in 0 1 2 3
    do
      python HIRA.py --dt 6.4e-12 --down_sample_time 1 \
            --verbose_every 100 --plot False \
            --filename ../../../git_ignore/ti-hira/ti_hira_v3.txt --center_x -.5 \
            --x1 .8 .9 -.4 .5 --x2 0 0 0 .6 --x3 0 0 .5 0 --find_center True --coeff_derivative 0 \
            --before .6 --phase_correction 0 0 0 .14 --t_max 2.8 \
            --n_points "$n_points" --n_tail "$n_points" \
            --order "$order" --scale 1e9 --shift 2 --seed $seed \
            > data/v7-ti-hira-order-"$order"-n_points-$n_points-seed-$seed.txt &
    done
  done
done

# python postprocessing.py --dt 6.4e-12 --down_sample_time 1 --scale 1e9 --case book --filename '../../../git_ignore/GLOBALEM/opt-result-Wed Apr 19 16:46:40 2023.csv_params.pickle'
