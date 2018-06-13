#PBS -N run_lsae
#PBS -l walltime=01:30:00
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

#name="test-zca-${PBS_ARRAYID}"
name="test-zca-100k-n400"
mkdir "logs/$name"

python -u sae.py -s $100000 --name="$name" >& "logs/${name}/train.log"
