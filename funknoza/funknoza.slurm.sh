#!/bin/bash -l

for order in {90..90..2}
do
	echo "#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=500G
#SBATCH --mail-user=elias.leboudec@epfl.ch
#SBATCH --mail-type=ALL
module purge
module load mathematica
wolframscript -script field_to_num.m a=4 order=$order nPoints=81" > .jobs/order-$order.job
	sbatch .jobs/order-$order.job
done
