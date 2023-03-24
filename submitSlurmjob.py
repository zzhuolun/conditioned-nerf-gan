import argparse
import os

# import sys
from datetime import datetime

# import json
# import configs
# from configs import curriculums

GPU = "a40"
GPU_NUM = 1
CPU_NUM = 6
CPU_MEM = "16G"
TIME = "1-00:00:00"
DDP = True if GPU_NUM > 1 else False
SUB_FOLDER = 'thesis'
# DATASET_SAMPLING_MODE = "sixthousand"

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run model")
    arg_parser.add_argument(
        "experimentDir",
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--sub_folder",
        type=str,
        default=SUB_FOLDER,
        help="subdirectory under slurm/zhouzh to save multiple experiments",
    )
    arg_parser.add_argument(
        "-s",
        dest="submitJob",
        action="store_true",
        help="Also submit job",
    )
    arg_parser.add_argument(
        "-t",
        dest="trackJob",
        action="store_true",
        help="Track job",
    )
    arg_parser.add_argument(
        "--edit",
        dest="editLog",
        action="store_true",
        help="Edit log instead of tail",
    )
    arg_parser.add_argument(
        "--nt", dest="donttrain", action="store_true", help="Don't train just test"
    )
    arg_parser.add_argument(
        "--cancel",
        dest="cancel",
        action="store_true",
        help="Cancel latest job",
    )
    arg_parser.add_argument(
        "-m",
        dest="message",
        type=str,
        default="",
        help="description of the submitted job/experiment, will be saved in under the experiment folder",
    )
    arg_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="As in train.py",
    )
    arg_parser.add_argument(
        "--config_base",
        type=str,
        default=None,
        help="As in train.py",
    )
    arg_parser.add_argument(
        "--stop_step",
        type=int,
        default=None,
        help="Stop training at stop_step.",
    )
    filename = "tempslurm.sh"
    args = arg_parser.parse_args()
    expDir = os.path.join("/storage/slurm/zhouzh",
                          args.sub_folder, args.experimentDir)
    print("Experiment folder: ", expDir)
    submitJob = bool(args.submitJob)
    trackJob = bool(args.trackJob)
    editLog = bool(args.editLog)
    donttrain = bool(args.donttrain)
    cancel = bool(args.cancel)

    expLogsDir = os.path.join(expDir, "logs")
    if not os.path.isdir(expLogsDir):
        os.makedirs(expLogsDir)
    if submitJob:
        # write slurm script
        with open(filename, "w+") as f:
            f.write("#!/bin/bash \n")
            f.write('#SBATCH --job-name="' + os.path.basename(expDir) + '" \n')
            f.write("#SBATCH --nodes=1 \n")
            f.write(f"#SBATCH --cpus-per-task={CPU_NUM} \n")
            f.write(f"#SBATCH --gpus={GPU}:{GPU_NUM}\n")
            f.write(f"#SBATCH --mem={CPU_MEM} \n")
            f.write(f"#SBATCH --time={TIME} \n")
            f.write("#SBATCH --mail-type=END,TIME_LIMIT\n")
            f.write("#SBATCH --partition=NORMAL \n")
            # f.write("#SBATCH --no-requeue \n")
            f.write("#SBATCH --output=" + expLogsDir + "/slurm-%j.out \n")
            f.write("#SBATCH --error=" + expLogsDir + "/slurm-%j.out \n")
            f.write("nvidia-smi \n")
            f.write("source /usr/stud/zhouzh/miniconda3/etc/profile.d/conda.sh\n")
            f.write("conda activate new-pigan\n")
            f.write("echo Starting job ${SLURM_JOBID}\n")
            f.write("squeue -j ${SLURM_JOBID} -O nodelist | tail -n +2\n")
            f.write('echo "Run started at `date`"\n')
            if not donttrain:
                TRAIN_CMD = f"srun python {expDir}/src/train.py --output_dir {expDir}"
                TRAIN_CMD += f" --config {args.config}" if args.config else ""
                TRAIN_CMD += f" --config_base {args.config_base}" if args.config_base else ""
                TRAIN_CMD += " --ddp" if DDP else ""
                TRAIN_CMD += f' --stop_step {args.stop_step}' if args.stop_step else ""
                f.write(TRAIN_CMD + "\n")
            f.write('echo "Run completed at `date`"\n')

        os.system("./version.sh")
        os.system("mv " + filename + " " + expDir + "/.")
        os.system(f"mv src" + " " + expDir + "/.")
        # os.system(f"tar --exclude='*/__pycache__' -cvzf {expDir}/src.tar configs datasets.py discriminators generators inference.py submitSlurmjob.py train.py utils.py fid_evaluation.py metric_utils.py version")
        os.system("sbatch " + expDir + "/" + filename)
        # save the description of the experiment
        if args.message:
            with open(os.path.join(expDir, "description"), "a+") as g:
                g.write(str(datetime.now()) + "\n")
                g.write(args.message + "\n")

    elif trackJob or cancel:
        files = os.listdir(expLogsDir)
        trackingFiles = []
        for file in files:
            if file.startswith("slurm-"):
                trackingFiles.append(file)
        trackingFiles = sorted(trackingFiles)
        trackFile = trackingFiles[-1]
        if cancel:
            os.system("scancel " + trackFile[6:-4])
            print("Cancelled " + trackFile[6:-4])
        else:
            print("Tracking: " + trackFile)
            if editLog:
                os.system("vim " + os.path.join(expLogsDir, trackFile))
            else:
                os.system("tail -20 " + os.path.join(expLogsDir, trackFile))
