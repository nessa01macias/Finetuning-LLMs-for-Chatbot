#!/bin/bash



#SBATCH --job-name=base-models-scores

#SBATCH --account=project_2008167

#SBATCH --partition=gpu

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=10

#SBATCH --mem=50G

#SBATCH --time=15:00:00

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

cd /scratch/project_2008167/thesis/evaluation/scores



# Execute the Python script

python scores-generation.py	


echo "------------------------------------------------"

echo "Script finished running"

