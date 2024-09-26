#!/bin/zsh
noise_level=10.0
for order in {1..1..2}
do
  for n_points in 45
  do
    for seed in {0..1}
    do
      python HIRA.py --dt 6.4e-12 --down_sample_time 1 \
            --verbose_every 100 --plot False \
            --filename ../../../../git_ignore/ti-hira/ti_hira_v3.txt --center_x -.5 \
            --x1 .8 .9 -.4 .5 --x2 0 0 0 .6 --x3 0 0 .5 0 --find_center True --coeff_derivative 0 \
            --before .6 --phase_correction 0 0 0 0 --t_max 2.8 \
            --n_points "$n_points" --n_tail "$n_points" --noise_level "$noise_level" \
            --order "$order" --scale 1e9 --shift 2 --seed $seed --save_path data \
            > ../../../../git_ignore/ira/opt_results//v8-ti-hira-order-"$order"-n_points-$n_points-seed-$seed-noise_level-"$noise_level".txt &
    done
  done
done

# SNR = inf:
# python postprocessing.py --dt 6.4e-12 --down_sample_time 1 --scale 1e9 --case book --filename '../../../git_ignore/GLOBALEM/opt-result-Sat Apr 22 13:36:54 2023.csv_params.pickle'

# SNR = 0 dB:
# python postprocessing.py --dt 6.4e-12 --down_sample_time 1 --scale 1e9 --case book --filename 'data/opt-result-Tue Jan  9 21:37:50 2024.csv_params.pickle' --dashed
