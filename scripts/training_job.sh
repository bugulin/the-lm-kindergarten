#!/bin/bash
#PBS -N npfl140_training
#PBS -l select=1:ncpus=2:ngpus=1:mem=16gb:gpu_mem=24gb:scratch_local=64gb
#PBS -l walltime=15:00:00

## JOB SETTINGS
REPOSITORY="https://github.com/bugulin/SemEval-2026-11"
BRANCH="main"
# arguments passed to src/cli.py
SCRIPT_ARGS="fine-tune --thinking --dataset data/1/train_data.json --output-repo Jajasek/llama-3.1-syllogism-grpo-lora"
# Optional: path to the file containing the huggingface token, relative to the directory from which the job is submitted
HF_TOKEN_PATH=".hf_token"

# Print out the basic info about the job
echo "Hello ${PBS_JOBNAME} at $(date) from user ${USER}!"
echo "${PBS_JOBID} is running on node $(hostname -f) in a scratch directory ${SCRATCHDIR}"
echo "Repository: $REPOSITORY"
echo "Branch: $BRANCH"
echo "Command: src/cli.py $SCRIPT_ARGS"

# Debug output - print all executed lines to stdout
set -x
# Exit on any error
set -e

trap 'clean_scratch' EXIT

outdir="out"

cd "${SCRATCHDIR}" || exit

# Prepare the environment
git clone --branch="${BRANCH}" --depth=1 "${REPOSITORY}" repo
cd repo || exit
curl -LsSf https://astral.sh/uv/install.sh | env UV_PRINT_QUIET=1 UV_UNMANAGED_INSTALL="bin" sh
mkdir -p "${outdir}"

# Authenticate with HuggingFace
hf_token_file="${PBS_O_WORKDIR}/${HF_TOKEN_PATH}"
if [[ -f "${hf_token_file}" ]]; then
    HF_TOKEN=$(cat "${hf_token_file}")
    export HF_TOKEN
fi

# Run the main task
bin/uv run src/cli.py fine-tune -o "${outdir}" "${FINE_TUNE_ARGS[@]}"

# Clean up
mv "${outdir}" "${PBS_O_WORKDIR}/${PBS_JOBNAME}.${PBS_JOBID}"
