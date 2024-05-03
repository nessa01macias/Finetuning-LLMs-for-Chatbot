#!/bin/bash



#SBATCH --job-name=base_model_phi2_2b-bleu-evaluation

#SBATCH --account=project_2008167

#SBATCH --partition=gpu

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=10

#SBATCH --mem=50G

#SBATCH --time=20:00:00

#SBATCH --gres=gpu:v100:2

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

cd /scratch/project_2008167/thesis/evaluation/base_models



# Execute the Python script

python phi2_2b.py



echo "------------------------------------------------"

echo "Script finished running"


