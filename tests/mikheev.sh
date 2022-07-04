for order in 0 1 2 3 4 5 6 7 8 9 10
do
 python mikheev.py --down_sample_time 1 --tol 1e-7 \
    --n_points 0 --n_tail 10 --verbose_every 50 --plot False --scale 1e5 --find_center False \
    --order $order --norm 4 \
    > data/order-$order-mikheev_v1.txt &
done