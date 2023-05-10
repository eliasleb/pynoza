#!/bin/bash -l

for order in {1..39..2}
do
	echo "#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
module purge
module load mathematica
wolframscript -script time_domain.m a=4 maxOrder=$order nPoints=31 $PWD/data" > .jobs/order-$order.job
	sbatch .jobs/order-$order.job
done
