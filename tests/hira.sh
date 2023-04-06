for order in 5
do
  for n_points in {30..150..5}
  do
    python HIRA.py --dt 6.4e-12 --down_sample_time 1 \
        --n_points "$n_points" --n_tail $(( n_points * 100 / 100 )) --verbose_every 100 --plot False \
        --filename ../../../git_ignore/ti-hira/ti_hira_v2.txt --center_x -.5 \
        --x1 .8 .9 -.4 .5 --x2 0 0 0 .6 --x3 0 0 .5 0 --find_center True --scale 1e-13 --coeff_derivative 0 \
        --before .6 --phase_correction 0 0 0 .15 \
        --t_max 2.8 --order $order > data/order-$order-ti-hira-$n_points-v7.txt &
    done
done
