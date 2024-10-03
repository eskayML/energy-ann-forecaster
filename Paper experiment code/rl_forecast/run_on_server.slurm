#!/bin/zsh
#SBATCH --job-name=econometrics_paper_forecasts
#SBATCH --output=logs/econmetrics_paper_%j.log
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=10
#SBATCH --partition=holon
#SBATCH --mem-per-cpu=3G 
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jhenze@uni-kassel.de
#SBATCH --hint=nomultithread
date;hostname;pwd


echo $SHELL



cd  /mnt/work/csells/paper-forecast-methods

source /mnt/work/csells/paper-forecast-methods/env/bin/activate

cd  /mnt/work/csells/paper-forecast-methods/load_forecast/rl_forecast


# Those environment variables can also be accessed within the python script. E.g.:
# SLURM_CPUS_PER_TASK = os.getenv('SLURM_CPUS_PER_TASK')
# if SLURM_CPUS_PER_TASK is not None:
#     # Set environment variables
#     os.environ['OMP_NUM_THREADS'] = SLURM_CPUS_PER_TASK

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK

#srun python experiment_file.py
srun -n1 -N1 --label python -u RL_Train_AE_full.py --drop_last -j $SLURM_CPUS_PER_TASK --loadae --skip1000s --skip-trained &
srun -n1 -N1 --label python -u RL_Train_AE_full.py --drop_last --reduced -j $SLURM_CPUS_PER_TASK --loadae --skip1000s --skip-trained &

wait
