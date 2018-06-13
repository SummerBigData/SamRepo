#PBS -N run_mnist
#PBS -l walltime=05:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=16GB
#PBS -j oe

if [ -z "$PBS_O_WORKDIR" ]
then
	echo "PBS_O_WORKDIR not defined"
else
	cd $PBS_O_WORKDIR
	echo $PBS_O_WORKDIR
fi

module load python/2.7.8

#name="h${PBS_ARRAYID}_L3_l1_s60000"
#name="h25_L3_l1_s${PBS_ARRAYID}"
#name="h25_L3_l${PBS_ARRAYID}_s60000"
name="h50_L3_l4_s60000"
mkdir "logs/$name"

python -u nn.py --size_h=50 --num_h=2 --lambda=3 --num_samp=60000 --max_iter=1000 --all_data --name="$name" >& "logs/${name}/train.log"
