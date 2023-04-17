#!/bin/zsh
for order in 1 3 5
do
  for n_points in 30 35 40
  do
    for phase_correction in .1 .12 .14 .16 .18 .2
    do
      python HIRA.py --dt 6.4e-12 --down_sample_time 1 \
            --verbose_every 100 --plot False \
            --filename ../../../git_ignore/ti-hira/ti_hira_v2.txt --center_x -.5 \
            --x1 .8 .9 -.4 .5 --x2 0 0 0 .6 --x3 0 0 .5 0 --find_center True --coeff_derivative 0 \
            --before .6 --phase_correction 0 0 0 $phase_correction --t_max 2.8 \
            --n_points "$n_points" --n_tail "$n_points" \
            --order $order --scale 1e-8 --shift 2 \
            > data/v1-ti-hira-order-$order-n_points-$n_points-phase-$phase_correction.txt &
        done
    done
done
