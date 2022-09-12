# Copyright (C) 2022  Elias Le Boudec
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Affero General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Affero General Public License for more details.
#
#    You should have received a copy of the GNU Affero General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

for order in 0 1 2 3 4 5 6 7 8 9 10
do
 python mikheev.py --down_sample_time 1 --tol 1e-8 \
    --n_points 10 --n_tail 10 --verbose_every 10 --plot False --scale 1e4 --find_center False \
    --order $order --norm 2 \
    > data/order-$order-mikheev_v5.txt &
done
