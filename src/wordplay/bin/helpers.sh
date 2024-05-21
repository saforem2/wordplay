#!/bin/bash --login
#
# set -euxo pipefail

if [[ -n "${PBS_O_WORKDIR}" ]]; then
    WORKING_DIR="${PBS_O_WORKDIR}"
elif [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
    WORKING_DIR="${SLURM_SUBMIT_DIR}"
else
    echo "Unable to detect PBS or SLURM working directory info..."
    WORKING_DIR=$(python3 -c 'import os; print(os.getcwd())')
    echo "Using ${WORKING_DIR} as working directory..."
fi

export WORKING_DIR="${WORKING_DIR}"
printf "Using WORKING_DIR: %s\n" ${WORKING_DIR}


function which_machine() {
    if [[ $(hostname) == x4* ]]; then
        local machine="aurora"
    elif [[ $(hostname) == x1* ]]; then
        local machine="sunspot"
    elif [[ $(hostname) == x3* ]]; then
        if [[ "${PBS_O_HOST}" == sirius* ]]; then
            local machine="sirius"
        else
            local machine="polaris"
        fi
    elif [[ $(hostname) == nid* ]]; then
        local machine="perlmutter"
    else
        # echo "Unknown MACHINE. Setting MACHINE to $(hostname) and continuing..."
        local machine=$(hostname)
    fi
    echo "${machine}"
}

function get_machine() {
    # if [[ $(hostname) == x4* ]]; then
    #     machine="aurora"
    # elif [[ $(hostname) == x1* ]]; then
    #     machine="sunspot"
    # elif [[ $(hostname) == x3* ]]; then
    #     if [[ "${PBS_O_HOST}" == sirius* ]]; then
    #         machine="sirius"
    #     else
    #         machine="polaris"
    #     fi
    # elif [[ $(hostname) == nid* ]]; then
    #     machine="perlmutter"
    # else
    #     echo "Unknown MACHINE. Setting MACHINE to $(hostname) and continuing..."
    # fi
    machine=$(which_machine)
    export MACHINE="${machine}"
    printf "Running on: %s\n" "$(printBlue ${MACHINE})"
}



function check_and_kill_if_running() {
    # kill $(ps aux | grep -E "$USER.+(mpi|main.py)" | grep -v grep | awk '{print $2}')
    RUNNING_PIDS=$(lsof -i:29500 -Fp | head -n 1 | sed 's/^p//')
    if [[ -n "${RUNNING_PIDS}" ]];
        then echo "Caught ${RUNNING_PIDS}" && kill "${RUNNING_PIDS}";
    else
        echo "Not currently running. Continuing!"
    fi
}


function setupSrun() {
    if [[ $(hostname) == login* || $(hostname) == nid* ]]; then
        export NHOSTS="${SLURM_NNODES:-1}"
        export NGPU_PER_HOST="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}"
        export NGPUS="$(( NHOSTS * NGPU_PER_HOST ))"
        export SRUN_EXEC="srun --gpus ${NGPUS} --gpus-per-node ${NGPU_PER_HOST} -N ${NHOSTS} -n ${NGPUS} -l -u --verbose"
    else
        echo "Skipping setupSrun() on $(hostname)"
    fi
}


function printJobInfo() {
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "- MPICH_DIR=${MPICH_DIR:-${MPI_ROOT}}"
    echo "- Using $(which python3)"
    echo "- WORLD_SIZE:${WORLD_SIZE}"
    echo "- NCCL: ${NCCL:-nccl}"
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++"
}

function setupLauncher() {
    # outdir=$1
    if [[ -n "${DIST_LAUNCH}" && ${LAUNCH_CMD:-"MPICH"} != "deepspeed" ]]; then
        export LAUNCH_CMD="${DIST_LAUNCH} --genvall --cpu-bind depth -d 16 $(which python3) -Wignore ${EXEC}"
    else
        # Assert `./hostfile_deepspeed` exists
        export hfds="${WORKING_DIR}/hostfile_deepspeed" && [ -f "${hfds}" ] || exit
        export LAUNCH_CMD="deepspeed --hostfile $hfds --launcher MPICH ${EXEC}"
    fi
    printf "%s" "$(printRed 'Launching with:')"
    printf " %s" "$(printMagenta ${LAUNCH_CMD})"
}

function setDSlauncher() {
    # launcher setting
    outdir=$1
    export hfds="$outdir/hostfile_deepspeed"
    export hfmpi="$outdir/hostfile_mpich"
    [ -f "$hfds" ] || exit
    [ -f "$hfmpi" ] || exit
    export LAUNCHER=${LAUNCHER:-MPICH}
    if [[ $LAUNCHER == "deepspeed" ]]; then
        export launcher=""
    else
        export launcher="--force_multi --hostfile $hfds --launcher=${LAUNCHER} --launcher_args='-hostfile ${hfmpi}'"
    fi
}


function make_ds_hostfile() {
    export GPUS_PER_NODE="${GPUS_PER_NODE:-${NGPU_PER_HOST:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}}}"
    # ---- Make MPICH hostfile ----------------
    hf="${HOSTFILE:-${PBS_NODEFILE}}"
    export hostfile_mpich=hostfile_mpich
    cat "${hf}" > "${hostfile_mpich}"
    # ---- Make DeepSpeed hostfile -------------------
    export hostfile_deepspeed=hostfile_deepspeed
    cat "${hf}" > "${hostfile_deepspeed}"
    sed -e "s/$/ slots=${GPUS_PER_NODE}/" -i "${hostfile_deepspeed}"
}

# +---------------------------------------+
# | 1. Git clone ezpz (if not found)    |
# | 2. Install ezpz (if not installed)  |
# +---------------------------------------+
function ezpz() {
    if [[ ! -d "${WORKING_DIR}/deps/ezpz" ]]; then
        mkdir -p "${WORKING_DIR}/deps"
        git clone https://github.com/saforem2/ezpz "${WORKING_DIR}/deps/ezpz"
    else
        echo "Found ezpz!"
    fi
    # echo "Done with clone. Now, checking if ezpz is installed..."
    # if python3 -c 'import ezpz; print(ezpz.__file__)' 2> '/dev/null'; then
    # if [[ $(python3 -c "import sys; any(['ezpz' in s for s in sys.path])") 2> '/dev/null' ]]; then
    #     echo "Has ezpz installed. Nothing to do."
    # else
    #     echo "Does not have ezpz installed. Installing..."
    #     echo "Using $(which python3) to install ezpz:"
    #     python3 -m pip install -e "${WORKING_DIR}/deps/ezpz" --verbose --require-virtualenv #  > ezpz-install.log 2>&1
    # fi
    if [[ $(python3 -c "import sys; any(['ezpz' in s for s in sys.path])") 2> '/dev/null' ]]; then
        echo "Installing ezpz in ${WORKING_DIR}/deps/ezpz"
        python3 -m pip install -e "${WORKING_DIR}/deps/ezpz" --verbose --require-virtualenv
    fi
    # echo "Done with ezpz."
    source ${WORKING_DIR}/deps/ezpz/src/ezpz/bin/savejobenv  >  /dev/null 2>&1 #> /tmp/savejobenv.log 2>&1 || exit
    source ${WORKING_DIR}/deps/ezpz/src/ezpz/bin/getjobenv || exit
    make_ds_hostfile || exit
}

# +------------------------------------------------------------------------+
# | Save important environment variables to .deepspeed_env, which will be  |
# | forwarded to ALL ranks with DeepSpeed                                  |
# +------------------------------------------------------------------------+
function saveDSenv() {
    echo "Saving {PATH, LD_LIBRARY_PATH, htt{p,ps}_proxy, CFLAGS, PYTHONUSERBASE} to .deepspeed_env"
    {
        echo "PATH=${PATH}" ;
        echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" ;
        echo "http_proxy=${http_proxy}" ;
        echo "https_proxy=${https_proxy}" ;
        echo "CFLAGS=${CFLAGS}" ;
        echo "PYTHONUSERBASE=$PYTHONUSERBASE" ;
    } > .deepspeed_env
}

function setOutput() {
    # ---- Specify output location --------------------------------
    OUTPUT_PREFIX="ws${WORLD_SIZE}_ds_stage${ZERO_STAGE}_nl${NLAYERS}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX}_hs${HIDDEN}_mb${MICRO_BATCH}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX}_seq${SEQ}_gb${GLOBAL_BATCH}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX}_pp${PP}_tp${TP}_${DTYPE}_opt${OPT}"
    OUTPUT_PREFIX="${OUTPUT_PREFIX}_lr${LR}_lwf${LR_WARMUP_FRAC}"
    if [[ -n "${LR_DECAY_ITERS}" ]]; then
        OUTPUT_PREFIX="${OUTPUT_PREFIX}_ldi${LR_DECAY_ITERS}"
    fi
    if [[ -z "${NO_FLASH_ATTN:-}" ]]; then
        OUTPUT_PREFIX="${OUTPUT_PREFIX}_flash"
    fi
    export OUTPUT_PREFIX="${OUTPUT_PREFIX}"
    # OUTPUT_DIR="logs/${OUTPUT_PREFIX}/$(date +%m%d%H%M%S)_${HOSTNAME}"
    OUTPUT_DIR="logs/${OUTPUT_PREFIX}/$(date +%Y%m%d-%H%M%S)_${WORLD_SIZE}_${HOSTNAME}"
    export OUTPUT_DIR="${OUTPUT_DIR}"
    export OUTPUT_LOG="${OUTPUT_DIR}/output.log"
    export CKPT_DIR="checkpoints/${OUTPUT_PREFIX}"
    echo "${OUTPUT_LOG}" >> "logs/latest"
    mkdir -p "${OUTPUT_DIR}"
    printf "\n Please see logs at: %s\n" $(printGreen "${OUTPUT_DIR}")
    printf "Checkpoints will be saved to: %s\n" $(printYellow "${CKPT_DIR}")
}

function buildDSconfig() {
    # ---- Build DeepSpeed Config ---------------------------------
    export CPU_OPTIMIZER="${CPU_OPTIMIZER:-0}"
    export DS_CONFIG="${WORKING_DIR}/ds-configs/ds_stage${ZERO_STAGE}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}_pp${PP}_${DTYPE}.json"
    mkdir -p $(dirname "${DS_CONFIG}")
    echo "DS_CONFIG: ${DS_CONFIG}"
    printf "ZS: %s, , MB: %s, GB: %s, PP: %s, DTYPE: %s" "${ZERO_STAGE}" "${CPU_OPTIMIZER}" "${MICRO_BATCH}" "${GLOBAL_BATCH}" "${PP}" "${DTYPE}"
    # working_dir="${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
    generateDSconfig "${DS_CONFIG}"
    # bash "${WORKING_DIR}/ALCF/generate_ds_config.sh" "${DS_CONFIG}"
    # -------------------------------------------------------------
}


function sumWeights() {
    local file_list=$1
    weights=$(cat "${file_list}" | awk '{print $1}' | tr '\n' '\ ,\ ' | sed 's/^/[/g' | sed 's/$/]/g' | tr '\ ' "\,\ ")
    python3 -c "import numpy as np; print(np.sum(${weights}))"
}

function sumFiles() {
    local rd=$1
    for f in $("${rd}/*.txt"); do
        ws=$(sumWeights "${rd}/${f}")
        echo "sum($f.weights)=${ws}"
    done
}

########################################################
# Setup / activate conda environment,
########################################################
setup_conda_sunspot() {
    ###### check if CONDA_PREFIX non-empty ################
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
        # eval "$(~/miniconda3/bin/conda shell.zsh hook)"
        # conda activate anl_24_q2_release
        module use /soft/preview-modulefiles/24.086.0 ; module load frameworks/2024.04.15.002.lua
    fi
    ###### check if VIRTUAL_ENV non-empty ####################################
    # venvs/anl_24_q2_release/bin/activate
    # if [[ -d "${DEFAULT_VENV_PATH}" ]]; then
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        if [[ -n "${CONDA_PREFIX}" ]]; then
            VENV_DIR="${WORKING_DIR}/venvs/$(echo ${CONDA_PREFIX} | tr '\/' '\t' | awk '{print $NF}')"
        else
            VENV_DIR="${WORKING_DIR}/venvs/anl_24_q2_release"
        fi
        echo "Caught virtual env at ${VENV_DIR}!"
        # source "${VENV_DIR}/bin/activate" || 
        if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
            printf "[!! %s]: Unable to locate %s\n" "$(printRed "ERROR")" "$(printMagenta "${VENV_DIR}/bin/activate")"
            # echo "[!ERROR]: Unable to locate ${VENV_DIR}/bin/activate !!"
        else
            source "${VENV_DIR}/bin/activate"
        fi
    else
        echo "Found existing python at: $(which python3)"
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
    setup_venv_from_conda
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

function setEnv() {
    local virtual_env="${VIRTUAL_ENV-}"
    local conda_prefix="${CONDA_PREFIX-}"
    if [[ -n "${conda_prefix}" && -z "${virtual_env}" ]]; then
        echo "No virtual environment found."
        echo "Using conda from: ${conda_prefix}"
        IN_CONDA=1
        IN_VENV=0
    elif [[ -n "${virtual_env}" && -z "${conda_prefix}" ]]; then
        echo "No conda found."
        echo "Using virtual_env from: ${virtual_env}"
        IN_CONDA=0
        IN_VENV=1
    elif [[ -n "${virtual_env}" && -n "${conda_prefix}" ]]; then
        echo "Using virtual_env: ${virtual_env} on top of conda from: ${conda_prefix}"
        IN_CONDA=1
        IN_VENV=1
    elif [[ -z "${conda_prefix}" && -z "${virtual_env}" ]]; then
        echo "No conda_prefix or virtual_env found in environment..."
        echo "Setting up conda..."
        ######################## setup_conda ############################
        # ---- [SunSpot @ ALCF]  || [Aurora @ ALCF] ---------------------
        if [[ $(hostname) == x1* || $(hostname) == x4* ]]; then
            # ----- [Aurora] --------------------------------------------
            if [[ -z "${conda_prefix}" && -z "${virtual_env}" ]]; then
                if [[ $(hostname) == x4* ]]; then
                    # TODO: Update once Aurora back online
                    eval "$(conda shell.zsh hook)" && conda activate anl_release_q4v2
                # ----- [SunSpot] ---------------------------------------
                elif [[ $(hostname) == x1* ]]; then
                    echo "Running on SunSpot !!"
                    setup_conda_sunspot
                fi
            fi
            # MPICH_MODULES=$(echo $LOADEDMODULES | tr ':' '\n' | grep mpich)
            # if [[ -z "${MPICH_MODULES" ]]; then
            #     source "${WORKING_DIR}/ALCF/sunspot-env.sh" || exit
            # else
            #     echo "Caught MPICH_MODULES: ${MPICH_MODULES}"
            # fi
        # ----- [Polaris @ ALCF] --------------------------------------------
        elif [[ $(hostname) == x3* ]]; then
            if [[ "${PBS_O_HOST}" == sirius* ]]; then
                echo "Running on Sirius !!"
                setup_conda_sirius
            else
                echo "Running on Polaris !!"
                # ---- [load conda] -------------------------------------
                setup_conda_polaris
            fi
        # ----- [Perlmutter @ NERSC] ----------------------------------------
        elif [[ $(hostname) == login* || $(hostname) == nid* ]]; then
            echo "Running on Perlmutter !!"
            module load pytorch
            source "${SLURM_SUBMIT_DIR}/venvs/perlmutter/pytorch-2.1.0-cu12/bin/activate"
        else # ------------------------------------- [Unknown] -------------------
            echo "Unknown hostname $(hostname)"
            exit 1
        fi
    else
        echo "Unable to setup python environment. Exiting"
        exit 1
    fi
    #####################################################################
    pystr="Using: $(which python3)"
    printf "[python] %s" "$(printMagenta ${pystr})"
    printf "\n"
    export "PYTHON_EXEC=$(which python3)"
}

######################################################################
# `makeHostiles`:
#     Detect if `HOSTFILE` set in active environment.
#         - If so, use this.
#         - Otherwise, make default HOSTFILEs from "${PBS_NODEFILE}"
######################################################################
function makeHostfiles() {
    if [[ -n "${HOSTFILE}" ]]; then
        printf "!! USING CUSTOM HOSTFILE FROM: %s"  "${HOSTFILE}"
    else
        make_ds_hostfile
    fi
}

function generateDSconfig() {
    for v in "$GLOBAL_BATCH" "$MICRO_BATCH" "$GRAD_ACC_STEPS" "$ZERO_STAGE" \
             "$PP" "$DTYPE"
    do
      if [ -z $v ]; then
        echo "Please export required envs before execute $0"
        exit 1
      fi
    done
    if [ $# -ne 1 ]; then
      echo "Usage: $0 config_file"
      exit 1
    fi
    # \"optimizer\": {
    #   \"type\": \"AdamW\",
    #   \"params\": {
    #     \"lr\": ${LR},
    #     \"beta1\": 0.9,
    #     \"beta2\": 0.95,
    #     \"eps\": 1e-5,
    #     \"weight_decay\": 1e-1
    #   }
    # },
    # \"scheduler\": {
    #   \"type\": \"WarmupLR\",
    #   \"params\": {
    #       \"warmup_min_lr\": 0.00003,
    #       \"warmup_max_lr\": 0.0003,
    #       \"warmup_num_steps\": 5000
    #   }
    # },
    extra=""
    common="\
        \"train_batch_size\": $GLOBAL_BATCH,
        \"train_micro_batch_size_per_gpu\": $MICRO_BATCH,
        \"steps_per_print\": 1,
        \"gradient_accumulation_steps\": $GRAD_ACC_STEPS,
        \"zero_allow_untested_optimizer\": true,
        \"gradient_clipping\": 1.0,
        \"activation_checkpointing\": {
          \"partition_activations\": true,
          \"contiguous_memory_optimization\": true
        },
        \"wall_clock_breakdown\": false,"
    flops_profiler="\
        \"flops_profiler\": {
          \"enabled\": true,
          \"profile_step\": 2,
          \"module_depth\": -1,
          \"top_modules\": 1,
          \"detailed\": true,
          \"output_file\": null
        }"
    if [[ $DTYPE == "bf16" ]]; then
    dtype="\
        \"communication_data_type\": \"bf16\",
        \"fp16\": {
          \"enabled\": false,
          \"loss_scale\": 0,
          \"loss_scale_window\": 1000,
          \"hysteresis\": 2,
          \"min_loss_scale\": 1
        },
        \"bfloat16\": {
          \"enabled\": true,
          \"loss_scale\": 1.0
        },"
    elif [[ $DTYPE == "fp16" ]]; then
    dtype="\
        \"communication_data_type\": \"fp16\",
        \"fp16\": {
          \"enabled\": true,
          \"loss_scale\": 0,
          \"loss_scale_window\": 1000,
          \"hysteresis\": 2,
          \"min_loss_scale\": 1
        },
        \"bfloat16\": {
          \"enabled\": false,
          \"loss_scale\": 1.0
        },"
    else
      dtype="\"communication_data_type\": \"fp32\","
    fi
    if [ $ZERO_STAGE == 3 ]; then
    zero="\
        \"zero_optimization\": {
          \"stage\": 3,
          \"reduce_scatter\": false,
          \"mics_shard_size\": 4,
          \"mics_hierarchical_params_gather\": true,
          \"stage3_max_live_parameters\": 3e9,
          \"stage3_max_reuse_distance\": 3e9,
          \"stage3_param_persistence_threshold\": 1e5,
          \"stage3_prefetch_bucket_size\": 5e7,
          \"contiguous_gradients\": true,
          \"overlap_comm\": true,
          \"reduce_bucket_size\": 90000000,
          \"sub_group_size\": 1e9,
          \"offload_optimizer\": {
            \"device\": \"none\",
            \"buffer_count\": 4,
            \"pipeline_read\": false,
            \"pipeline_write\": false,
            \"pin_memory\": true
          }
        },"
    # elif [[ $ZERO_STAGE == 2 ]]; then
    elif [ "${ZERO_STAGE}" == 2 ] || [ "${ZERO_STAGE}" == 1 ]; then
    # if [[ -n "${CPU_OPTIMIZER}" ]]; then
    if [[ "${CPU_OPTIMIZER}" != 0 ]]; then
    echo "!!!! CAUGHT CPU_OPTIMIZER !!!!"
    zero="\
        \"zero_optimization\": {
            \"stage\": $ZERO_STAGE,
            \"offload_optimizer\": {
              \"device\": \"cpu\"
            }
        },"
    else
    zero="\
        \"zero_optimization\": {
          \"stage\": $ZERO_STAGE
        },"
    fi
    # elif [[ $ZERO_STAGE == 1 ]]; then
    if [[ $PP > 1 ]]; then
      extra="\
          \"data_types\": {
            \"grad_accum_dtype\": \"fp32\"
          },
          \"comms_logger\": {
            \"enabled\": true,
            \"verbose\": false,
            \"prof_all\": true,
            \"debug\": false
          },"
    else
      # echo 'please add the config for zero_stage 1 without pipeline-parallelism'
      extra="\
          \"comms_logger\": {
            \"enabled\": true,
            \"verbose\": false,
            \"prof_all\": true,
            \"debug\": false
          },"
    fi
    else
      echo 'Please add the correct config set!!!'
    fi
# flops_profiler must at the end because no ',' is allowed at the end
cat <<EOT > $1
{
$common
$zero
$dtype
$extra
$flops_profiler
}
EOT
}

function printBlack() {
    printf "\e[1;30m%s\e[0m\n" "$@"
}

function printRed() {
    printf "\e[1;31m%s\e[0m\n" "$@"
}

function printGreen() {
    printf "\e[1;32m%s\e[0m\n" "$@"
}

function printYellow() {
    printf "\e[1;33m%s\e[0m\n" "$@"
}

function printBlue() {
    printf "\e[1;34m%s\e[0m\n" "$@"
}

function printMagenta() {
    printf "\e[1;35m%s\e[0m\n" "$@"
}

function printCyan() {
    printf "\e[1;36m%s\e[0m\n" "$@"
}

function printWhite() {
    printf "\e[1;37m%s\e[0m\n" "$@"
}
