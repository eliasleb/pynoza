#!/bin/zsh
for order in 0 1 2 3 4 5
do
  for n_points in 45
  do
    for phase_correction in .14 .15 .16
    do
      python HIRA.py --dt 6.4e-12 --down_sample_time 1 \
            --verbose_every 100 --plot False \
            --filename ../../../git_ignore/ti-hira/ti_hira_v3.txt --center_x -.5 \
            --x1 .8 .9 -.4 .5 --x2 0 0 0 .6 --x3 0 0 .5 0 --find_center True --coeff_derivative 0 \
            --before .6 --phase_correction 0 0 0 $phase_correction --t_max 2.8 \
            --n_points "$n_points" --n_tail "$n_points" \
            --order $order --scale 1e9 --shift 2 \
            > data/v5-ti-hira-order-$order-n_points-$n_points-phase-$phase_correction.txt &
        done
    done
done

# python postprocessing.py --dt 6.4e-12 --down_sample_time 1 --scale 1e-8 --case book --filename '../../../git_ignore/GLOBALEM/opt-result-Mon Apr 17 14:45:46 2023.csv_params.pickle'
