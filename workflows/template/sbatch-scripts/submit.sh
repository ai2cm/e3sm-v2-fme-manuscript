#!/bin/bash -l

#SBATCH -A m4331_g
#SBATCH -q regular
#SBATCH -C gpu_hbm40g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH -t 01:00:00
#SBATCH --output=joblogs/%x-%j.out

# env variables
export ENDPOINT_HOST=$(hostname)
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=32

while [[ "$#" -gt 0 ]]
do case $1 in
    -c|--configdir) CONFIGDIR="${2%/}"
    shift;;
    -d|--datadir) DATADIR="${2%/}"
    shift;;
    -g|--group) GROUP="$2"
    shift;;
    -i|--image) IMAGE="$2"
    shift;;
    -n|--name) NAME="$2"
    shift;;
    -t|--jobtype) JOBTYPE="$2"
    shift;;
    -x|--ckptdir) CKPTDIR="${2%/}"
    shift;;
    *) echo "Unknown parameter passed: ${1}"
    exit 1;;
esac
shift
done

if [[ -z ${JOBTYPE} ]]
then
    JOBTYPE=train-and-inference
fi

if [[ -z "${CONFIGDIR}" ]]
then
    echo "Option -c,--configdir missing"
    exit 1;
elif [[ -z "${DATADIR}" ]]
then
    echo "Option -d, --datadir missing"
    exit 1;
elif [[ -z "${IMAGE}" ]]
then
    echo "Option -i, --image missing"
    exit 1;
elif [[ -z "${NAME}" ]]
then
    echo "Option -n, --name missing"
    exit 1;
fi

OUTDIR=${DATADIR}/output/${GROUP}/${NAME}

# directories to mount
VCONFIG="${CONFIGDIR}:/configmount"
VTRAIN="${DATADIR}/traindata:/traindata"
VVALID="${DATADIR}/validdata:/validdata"
VSTATS="${DATADIR}/statsdata:/statsdata"
VPRED="${DATADIR}/predictiondata:/predictiondata"
VOUT="${OUTDIR}:/output"

# NOTE: $CKPTDIR is only used for inference and prediction-data jobtypes and
# should point to an existing training_checkpoints directory
if [[ -z "${CKPTDIR}" ]]
then
    VCKPT="${OUTDIR}/training_checkpoints:/training_checkpoints"
else
    VCKPT="${CKPTDIR}:/training_checkpoints"
fi

# wandb config
export WANDB_API_KEY=$(cat ~/.config/wandb/api)
export WANDB_NAME=${NAME}
export WANDB_RUN_GROUP=${GROUP}

set -xe

mkdir -p $OUTDIR

case $JOBTYPE in
train)
    srun -u podman-hpc run --rm --gpu --net host --ipc host \
        -w $(pwd) -v $(pwd):$(pwd) \
        -v $VCONFIG -v $VTRAIN -v $VVALID -v $VSTATS -v $VOUT \
        --env 'ENDPOINT_HOST' \
        --env 'NCCL*' --env 'SLURM*' --env 'WANDB*' \
        --env 'HDF5_USE_FILE_LOCKING' \
        --env 'OMP_NUM_THREADS' \
        $IMAGE bash train.sh
    ;;
train-and-inference)
    srun -u podman-hpc run --rm --gpu --net host --ipc host \
        -w $(pwd) -v $(pwd):$(pwd) \
        -v $VCONFIG -v $VTRAIN -v $VVALID -v $VSTATS -v $VOUT \
        --env 'NCCL*' --env 'SLURM*' --env 'WANDB*' \
        --env 'HDF5_USE_FILE_LOCKING' \
        --env 'OMP_NUM_THREADS' \
        $IMAGE bash train-and-inference.sh
    ;;
inference)
    podman-hpc run --rm --gpu --net host --ipc host \
        -w $(pwd) -v $(pwd):$(pwd) \
        -v $VCONFIG -v $VVALID -v $VOUT -v $VCKPT \
        --env 'NCCL*' --env 'SLURM*' --env 'WANDB*' \
        --env 'HDF5_USE_FILE_LOCKING' \
        --env 'OMP_NUM_THREADS' \
        $IMAGE bash inference.sh
    ;;
prediction-data)
    podman-hpc run --rm --gpu --net host --ipc host \
        -w $(pwd) -v $(pwd):$(pwd) \
        -v $VCONFIG -v $VVALID -v $VPRED -v $VCKPT -v $VOUT \
        --env 'NCCL*' --env 'SLURM*' --env 'WANDB*' \
        --env 'HDF5_USE_FILE_LOCKING' \
        --env 'OMP_NUM_THREADS' \
        $IMAGE bash inference.sh
    ;;
*)
    echo "Unknown jobtype (-t, --jobtype) passed: ${JOBTYPE}"
    exit 1
    ;;
esac
