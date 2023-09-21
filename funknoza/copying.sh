#!/bin/zsh

for order in {90..90..2}
do
#	scp data/field-v2-order-"$order"-a-4.mx eleboude@jed.epfl.ch:/scratch/eleboude/pynoza/funknoza/data/field-v2-order-"$order"-a-4.mx
	scp eleboude@jed.epfl.ch:/scratch/eleboude/pynoza/funknoza/data/v2-order-"$order"-a-4.mx data
done
