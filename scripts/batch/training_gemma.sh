#!/bin/bash

#SBATCH --job-name=training_gemma

#SBATCH --account=project_2008167

#SBATCH --partition=gpu

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=10

#SBATCH --mem=150G

#SBATCH --time=06:00:00

#SBATCH --gres=gpu:v100:1

#SBATCH --mail-type=END,FAIL

#SBATCH --mail-user=melanym@metropolia.fi



echo "Starting the data processing script"

echo "------------------------------------------------"



# Load necessary modules

module load python-data

module load cuda/11.7.0



# Activate your Python virtual environment

source /scratch/project_2008167/venv/bin/activate



# Navigate to the directory containing your script

cd /scratch/project_2008167/thesis



# Execute the Python script

python training_gemma.py



echo "------------------------------------------------"

echo "Script finished running"


