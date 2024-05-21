#!/bin/bash --login

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
HOST=$(hostname)

# Resolve path to current file
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
    DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
    SOURCE=$(readlink "$SOURCE")
    [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
PARENT=$(dirname "${DIR}")
GRANDPARENT=$(dirname "${PARENT}")
ROOT=$(dirname "${GRANDPARENT}")
MAIN="$PARENT/__main__.py"
SETUP_SCRIPT="$DIR/setup.sh"
TRAIN_SCRIPT="$DIR/train.sh"
NCPUS=$(getconf _NPROCESSORS_ONLN)


function join_by {
    local d=${1-} f=${2-};
    if shift 2; then
        printf %s "$f" "${@/#/$d}"
    fi
}

# function setupVenv() {
#     VENV_DIR="$1"
#     if [[ -d "${VENV_DIR}" ]]; then
#         echo "Found venv at: ${VENV_DIR}"
#         source "${VENV_DIR}/bin/activate"
#     else
#         echo "Skipping setupVenv() on $(hostname)"
#     fi
# }
#
# function loadCondaEnv() {
#     if [[ "${CONDA_EXE}" ]]; then
#         echo "Already inside ${CONDA_EXE}, exiting!"
#     else
#         MODULE_STR="$1"
#         module load "conda/${MODULE_STR}"
#         conda activate base
#     fi
# }
#
# function setupPython() {
#     local conda_date=$1
#     local venv_path=$2
#     if [[ "${CONDA_EXE}" ]]; then
#         echo "Caught CONDA_EXE: ${CONDA_EXE}"
#     else
#         loadCondaEnv "${conda_date}"
#     fi
#     if [[ "${VIRTUAL_ENV}" ]]; then
#         echo "Caught VIRTUAL_ENV: ${VIRTUAL_ENV}"
#     else
#         setupVenv "${venv_path}"
#     fi
# }
#
#
setup_conda_sunspot() {
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        module use /soft/preview-modulefiles/24.086.0 ; module load frameworks/2024.04.15.002.lua
    fi
}


########################################################
# Setup / activate conda environment,
########################################################
setup_conda_sunspot() {
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        # module load frameworks/2023.12.15.002
        module use /soft/preview-modulefiles/24.086.0 ; module load frameworks/2024.04.15.002.lua
    fi
}

# ┏━━━━━━━┓
# ┃ NERSC ┃
# ┗━━━━━━━┛
setupPerlmutter() {
    if [[ $(hostname) == login* || $(hostname) == nid* ]]; then
        export MACHINE="Perlmutter"
        SLURM_NODES=$(scontrol show hostname "${SLURM_NODELIST}")
        SLURM_NODEFILE="${HOME}/.slurm-nodefile"
        printf "%s\n" "${SLURM_NODES[@]}" > "${SLURM_NODEFILE}"
        export HOSTFILE="${HOSTFILE:-${SLURM_NODEFILE}}"
        [ "$SLURM_JOB_ID" ] \
            && echo "Caught SLURM_JOB_ID: ${SLURM_JOB_ID}" \
            || echo "!!!!!! Running without SLURM allocation !!!!!!!!"
        module load libfabric cudatoolkit pytorch/2.0.1
        export NODELIST="${SLURM_NODELIST:-$(hostname)}"
        export NHOSTS="${SLURM_NNODES:-1}"
        export NGPU_PER_HOST="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
        export NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
        LAUNCH="srun -N ${NHOSTS} -n ${NGPUS} -l u"
    else
        echo "[setupPerlmutter]: Unexpected hostname $(hostname)"
    fi
}

########################
# Setup conda on Sirius
########################
setup_conda_sirius() {
    if [[ -z "${CONDA_PREFIX-}" && -z "${VIRTUAL_ENV-}" ]]; then
        export MAMBA_ROOT_PREFIX=/lus/tegu/projects/PolarisAT/foremans/micromamba
        shell_name=$(echo "${SHELL}" | tr "\/" "\t" | awk '{print $NF}')
        eval "$("${MAMBA_ROOT_PREFIX}/bin/micromamba" shell hook --shell ${shell_name})"
        micromamba activate 2024-04-23
    else
        echo "Found existing python at: $(which python3)"
    fi
}

########################
# Setup conda on Polaris
########################
setup_conda_polaris() {
    # unset MPICH_GPU_SUPPORT_ENABLED
    if [[ -z "${CONDA_PREFIX-}" ]]; then
        module use /soft/modulefiles ; module load conda/2024-04-29 ; conda activate base
    else
        echo "Caught CONDA_PREFIX=${CONDA_PREFIX}"
    fi
    # setup_venv_from_conda
}


setup_venv_from_conda() {
    if [[ -z "${CONDA_PREFIX}" ]]; then
        echo "No ${CONDA_PREFIX} found."  #  Exiting."
        # exit 1
    else
        if [[ -n "${VIRTUAL_ENV}" ]]; then
            echo "Already inside virtual env at ${VENV_DIR}!"
        elif [[ -z "${VIRTUAL_ENV}" ]]; then
            echo "No VIRTUAL_ENV found in environment!"
            echo "    - Trying to setup from ${CONDA_PREFIX}"
            CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
            VENV_DIR="${WORKING_DIR}/venvs/${CONDA_NAME}"
            echo "    - Using VENV_DIR=${VENV_DIR}"
            # VENV_DIR="venvs/$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')"
            # VENV_DIR="${WORKING_DIR}/venvs/$(echo ${CONDA_PREFIX} | tr '\/' '\t' | awk '{print $NF}')"
            # VENV_DIR="${WORKING_DIR}/venvs/anl_24_q2_release"
            # if [[ -f "${VENV_DIR}/bin/activate" ]]; then
            if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
                printf "\n    - Creating a new virtual env on top of %s in %s" "$(printBlue "${CONDA_NAME}")" "$(printGreen "${VENV_DIR}")"
                mkdir -p "${VENV_DIR}"
                python3 -m venv "${VENV_DIR}" --system-site-packages
                source "${VENV_DIR}/bin/activate" || exit
            elif [[ -f "${VENV_DIR}/bin/activate" ]]; then
                echo "    - Found existing venv, activating from $(printBlue "${VENV_DIR}")"
                source "${VENV_DIR}/bin/activate"
            else
                printf "\n    [!! %s]: Unable to locate %s\n" "$(printRed "ERROR")" "$(printMagenta "${VENV_DIR}/bin/activate")"
            fi
        fi
        # else
        #     printf "[!! %s]: Unable to locate %s\n" "$(printRed "ERROR")" "$(printMagenta "${VENV_DIR}/bin/activate")"
    fi

}


function setup_env() {
    # machine one of:
    # [aurora, polaris, sunspot, sirius, perlmutter, ...]
    local machine=$(get_machine)
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        if [[ "${machine}" == "polaris" ]]; then
            seutp_conda_polaris
        elif [[ "${machine}" == "sirius" ]]; then
            setup_conda_sirius
        elif [[ "${machine}" == "sunspot" ]]; then
            setup_conda_sunspot
        elif [[ "${machine}" == "aurora" ]]; then
            setup_conda_aurora
        fi
    fi
    if  [[ -z "${VIRTUA_ENV}" ]]; then
        setup_venv_from_conda
    fi
}


function setupLogs() {
    LOGDIR="${PARENT}/logs"
    LOGFILE="${LOGDIR}/${TSTAMP}-${HOST}_ngpu${NGPUS}_ncpu${NCPUS}.log"
    export LOGDIR="${LOGDIR}"
    export LOGFILE=$LOGFILE
    if [ ! -d "${LOGDIR}" ]; then
        mkdir -p "${LOGDIR}"
    fi
    echo "Writing to logfile: ${LOGFILE}"
    # Keep track of latest logfile for easy access
    echo "$LOGFILE" >> "${LOGDIR}/latest"
}

function pprint() {
    OUTPUT="$*"
    echo "┌─────────────────────────────────────────────────────────────────────┐"
    echo "│ [setup.sh]:  ${OUTPUT[*]}"
    echo "└─────────────────────────────────────────────────────────────────────┘"
}


function printJobInfo() {
    ARGS=$*
    HOSTS_ARR=$(cat "${HOSTFILE:-$(hostname)}")
    # if [[ "${HOSTFILE}" ]]; then
    #     HOSTS_ARR=$(cat "${HOSTFILE}")
    #     HOSTS=$(join_by ' ' "${HOSTS_ARR}")
    # elif [[ "${SLURM_NODELIST}" ]]; then
    #     HOSTS="${SLURM_NODELIST}"
    # else
    #     echo "[printJobInfo][WARNING]: HOSTFILE not set, using resources on localhost!"
    #     HOSTS=$(hostname)
    # fi
    echo "┌─────────────────────────────────────────────────────────────────────"  #┐"
    echo "│ [setup.sh]: Job started at: ${TSTAMP} on ${HOST} by ${USER}"
    echo "│ [setup.sh]: Job running in: ${DIR}"
    echo "└─────────────────────────────────────────────────────────────────────"  #┘"
    echo "┌─────────────────────────────────────────────────────────────────────"  #┐"
    echo "│ [setup.sh]: DIR=${DIR}"
    echo "│ [setup.sh]: MAIN=${MAIN}"
    echo "│ [setup.sh]: SETUP_SCRIPT=${SETUP_SCRIPT}"
    echo "│ [setup.sh]: TRAIN_SCRIPT=${TRAIN_SCRIPT}"
    echo "│ [setup.sh]: PARENT=${PARENT}"
    echo "│ [setup.sh]: ROOT=${ROOT}"
    echo "│ [setup.sh]: LOGDIR=${LOGDIR}"
    echo "│ [setup.sh]: LOGFILE=${LOGFILE}"
    echo "└─────────────────────────────────────────────────────────────────────"  #┘"
    # echo "┌─────────────────────────────────────────────────────────────────────"  # ┐"
    # echo "│ [setup.sh]: [Hosts][${HOSTFILE}]: "
    # echo "│ [setup.sh]:   ${HOSTS[*]}"
    # echo "└──────────────────────────────────────────────────────────────────┘"
    echo "┌─────────────────────────────────────────────────────────────────────"  # ┐"
    echo "│ [setup.sh]: HOSTS: "
    echo "│   $(join_by ', ' $HOSTS_ARR)"
    echo "│ [setup.sh]: Using ${NHOSTS} hosts from ${HOSTFILE}"
    echo "│ [setup.sh]: With ${NGPU_PER_HOST} GPUs per host"
    echo "│ [setup.sh]: For a total of: ${NGPUS} GPUs"
    echo "└─────────────────────────────────────────────────────────────────────"  #┘"
    echo "┌─────────────────────────────────────────────────────────────────────"  #┐"
    echo "│ [setup.sh]: Using python: $(which python3)"
    echo "│ [setup.sh]: ARGS: ${ARGS[*]}"
    echo "│ [setup.sh]: LAUNCH: ${LAUNCH} python3 ${MAIN} ${ARGS[*]}"
    echo "└─────────────────────────────────────────────────────────────────────"  #┘"
    echo "┌─────────────────────────────────────────────────────────────────────"  #┐"
    echo "│ [setup.sh]: Writing logs to ${LOGFILE}"
    echo '│ [setup.sh]: To view output: `tail -f $(tail -1 logs/latest)`'  # noqa
    echo "│ [setup.sh]: Latest logfile: $(tail -1 ./logs/latest)"
    echo "│ [setup.sh]: tail -f $(tail -1 logs/latest)"
    echo "└─────────────────────────────────────────────────────────────────────"  #┘"
    echo -e "\n"
}

function setupJob() {
    # if [[ $(hostname) == x3* ]]; then
    #     setupPolaris
    # elif [[ $(hostname) == thetagpu* ]]; then
    #     export DISABLE_PYMODULE_LOG=1
    #     setupThetaGPU
    # elif [[ $(hostname) == nid* || $(hostname) == login* ]]; then
    #     setupPelmutter
    #     HOSTS="${SLURM_NODELIST}"
    # else
    #     echo "[setupJob]: Unexpected hostname $(hostname)"
    #     exit 1
    # fi
    setup_env
    export NHOSTS
    export NGPU_PER_HOST
    export NGPUS
    export HOSTFILE
    export LAUNCH
    setupLogs
}
