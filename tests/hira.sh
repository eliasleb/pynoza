for order in 5
do
  for n_points in 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120
  do
    python HIRA.py --dt 6.4e-12 --down_sample_time 1 \
      --n_points $n_points  --n_tail $n_points --verbose_every 50 --plot False  \
      --filename ../../../git_ignore/ti-hira/ti_hira_v2.txt --center_x -.5 \
      --x1 .5 .9 .5 .5 0 --x2 0 0 .6 0 0 --x3 0 0 0 .6 .6 --find_center True --scale 1e-12 --coeff_derivative 0 \
      --before .6 \
      --order $order > data/order-$order-ti-hira-$n_points-v5.txt &
  done
done
