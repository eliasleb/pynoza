#!/bin/zsh

cd ..

for order in {24..30..2}
do
  python tests/test_EPFL_logo.py --case disc --method python --order "$order" &
done
