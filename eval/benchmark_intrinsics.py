import argparse
import os
import subprocess
import datetime
import re
import json
from pprint import pprint
import shutil
"""
This script is intended to be run from the root level of this repostitory. 
Please see the comments in the 'parse_user_args' for instructions on how to run.
"""


def parse_user_args():
    """
    Parse the users arguments. There are three modes of execution:
        (1) Generation mode ('-g'). Where embeddings are generated via a 
        parameter sweep speified in the config file ('-c'). The embeddings
        are generated to the 'outdir' folder specifed by the config file.
        Inside of this folder a folder called 'output_<datetime>'. Inside of 
        the 'outputs_<datetime>' folder a log folder contains the output for 
        the generation of each embedding and a folder named the same thing as
        the original embedding filename specifed in the config ("filenames")
        holds all the generated embeddings for said input.
        (2) Evaluation mode ('-e'). This evaluates the output of the generation 
        mode phase. Here the folder holding the generated embeddings is passed 
        in (-f) and every file in that folder is run across the intrinsic tasks.
        The output is a csv file called 'out.csv' that is placed 
        in the 'outdir' specified by the config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g", "--gen", action="store_true", help="Flag to generate embeddings.")
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        help="Configuration file for generating embeddings.")
    parser.add_argument(
        "-e",
        "--eval",
        action="store_true",
        help="Flag to evaluate embeddings.")
    parser.add_argument(
        "-f",
        "--folder",
        action="store",
        help="Folder holding the emebeddings to be evaluated.")
    args = parser.parse_args()
    return args


def download_intrinsics():
    """
    Downloads the datasets needed to run intrinsic tasks.
    """
    if not os.path.isdir("eval/intrinsic/testsets"):
        os.chdir("eval/intrinsic")
        os.system("bash download_data.sh")
        os.chdir("../..")


def read_config(filename):
    """
    Parses 'filename' which should be a json config containing the parameters
    to be swept. Returns them in a 'data' JSON object
    """
    with open(filename) as f:
        data = json.load(f)
    pprint(data)
    return data


def poll_processes(processes, num_cores, state=None):
    """
    Manages our processes. If we have less processes then more continue on and
    kick off more. If we have more processes than cores (or equal), loop over 
    all processes polling until one finishes. When one (or more) finishes (at 
    the same time) delete it from the processes list and move on.
    """
    if len(processes) < num_cores:
        return

    found = False
    delete_indexes = []
    while not found:
        for i in range(len(processes)):
            poll = processes[i].poll()
            if poll != None:
                processes[i].update_state(state)
                found = True
                delete_indexes.append(i)
    for i in range(len(delete_indexes)):
        del processes[delete_indexes[i] - i]


def wait_processes(processes, state=None):
    """
    Wait until all processes are finished. Usually called at the very end.
    """
    for proc in processes:
        proc.wait()
        proc.update_state(state)


def gen_embedddings(config, args):
    """
    Main method for generating embeddings via parameter sweep specified in 
    'config'.
    """

    def get_outdir_and_logdir(config, args):
        """
        Creates the output folder and log folder. Returns the corresponding paths
        for each folder.
        """
        outdir = config["outdir"]
        now = datetime.datetime.now()
        output_folder = os.path.join(outdir, "outputs") + "_m" + str(now.month) + "-d" \
                        + str(now.day) + "-h" + str(now.hour) + "-m" + str(now.minute)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        logdir = os.path.join(output_folder, "log")
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        # Save the configuration file in the output folder for this run.
        shutil.copyfile(args.config, os.path.join(output_folder, "config.json"))
        return output_folder, logdir

    def is_invalid_config(rb, r, cb, c):
        """
        Do not run kmeans if the number of row or column buckets is 1 or -1.
        -1 means 'num_rows' or 'num_cols' respectively. Running kmeans is 
        useless to bucket in these scenarios.
        """
        return (rb == "kmeans" and (r == -1 or r == 1)) or \
               (cb == "kmeans" and (c == -1 or c == 1)) or \
               (c == -1 and r == -1)

    def get_gen_cmd(filename, r, c, sweep, s, q, rb, outdir, cb, q_dim_r, q_dim_c, rots):
        cmd = f"python main.py" + \
                    f" -f {filename}" + \
                    f" --num_row_buckets {r}" + \
                    f" --num_col_buckets {c}" + \
                    f" --{sweep} {s}" + \
                    f" --quantizer {q}" + \
                    f" --row_bucketer {rb}" + \
                    f" --output_folder {outdir}" + \
                    f" --col_bucketer {cb}" + \
                    f" --quant_num_rows {q_dim_r}" + \
                    f" --quant_num_cols {q_dim_c}" + \
                    f" --rotation {rots}"
        log_file = os.path.join(logdir,
                                f"q{q}_r{rb}{r}_c{cb}{c}_{sweep}{s}.log")
        return cmd + f" 2>&1 | tee {log_file}"

    class GenProcess:

        def __init__(self, p):
            self.p = p

        def poll(self):
            return self.p.poll()

        def wait(self):
            self.p.wait()

        def update_state(self, state=None):
            return

    outdir, logdir = get_outdir_and_logdir(config, args)

    filename = config["filename"]
    processes = []
    for d in range(0,len(config["quant_num_rows"])):
        q_dim_r = config["quant_num_rows"][d]
        q_dim_c = config["quant_num_cols"][d]
        for rots in config["rotation"]:
            for q in config["quantizers"]:
                for cb in config["bucketers"]:
                    for rb in config["bucketers"]:
                        sweep = "num_centroids" if q == "kmeans" else "num_bits"
                        for r in config["num_row_buckets"]:
                            for c in config["num_col_buckets"]:
                                if is_invalid_config(rb, r, cb, c):
                                    break
                                for s in config[sweep]:
                                    cmd = get_gen_cmd(filename, r, c, sweep, s, q, rb,
                                                      outdir, cb, q_dim_r, q_dim_c, rots)
                                    print(cmd)
                                    proc = subprocess.Popen(
                                        cmd,
                                        shell=True)    #, stdout=subprocess.DEVNULL)
                                    processes.append(GenProcess(proc))
                                    poll_processes(processes, config["num_cores"])

    wait_processes(processes)


def eval_embeddings(config, args):
    """
    Main method to evaluate the embeddings.
    """

    def get_eval_cmd(task, task_class, filename):
        if task_class == "ws":
            return f"python ws_eval.py GLOVE {filename} testsets/{task_class}/{task}.txt"
        elif task_class == "analogy":
            return f"python analogy_eval.py GLOVE {filename} testsets/{task_class}/{task}.txt"
        raise ValueError("Task class not recognized.")

    class EvalProcess:

        def __init__(self, p, task_class, task, filename, cmd):
            self.p = p
            self.task = task
            self.task_class = task_class
            self.filename = filename
            self.cmd = cmd

        def poll(self):
            return self.p.poll()

        def wait(self):
            self.p.wait()

        def update_state(self, states):
            output = self.p.stdout.read().strip().decode("utf-8").split(" ")
            print(output)
            if self.task_class == "ws":
                try:
                    states[self.filename][self.task] = float(output[len(output)-1]) 
                except Error:
                    states[self.filename][self.task] = 0.0
            elif self.task_class == "analogy":
                try:
                    states[self.filename][self.task + "_add" ] = float(output[len(output)-2])
                    states[self.filename][self.task + "_mul"] = float(output[len(output)-1])
                except Error:
                    states[self.filename][self.task + "_add"] = float(0.0)
                    states[self.filename][self.task + "_mul"] = float(0.0)

    cwd = os.getcwd()
    os.chdir("eval/intrinsic/")

    # Holds the state.
    states = {}
    processes = []

    # Run the baselines first.
    states["baseline"] = {}
    for task_class in config["tasks"]:
        for task in config["tasks"][task_class]:
            cmd = get_eval_cmd(task, task_class, config["filename"])
            print(cmd)
            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                close_fds=True)
            processes.append(
                EvalProcess(proc, task_class, task, "baseline", cmd))
            poll_processes(processes, config["num_cores"], states)

    # Now run all the other generated embeddings.
    for filename in os.listdir(args.folder):
        states[filename] = {}
        for task_class in config["tasks"]:
            for task in config["tasks"][task_class]:
                cmd = get_eval_cmd(task, task_class,
                                   os.path.join(args.folder, filename))
                print(cmd)
                proc = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    close_fds=True)
                processes.append(
                    EvalProcess(proc, task_class, task, filename, cmd))

                poll_processes(processes, config["num_cores"], states)

    wait_processes(processes, states)
    
    """-------------------Write the output to a CSV.-------------------------"""
    head, tail = os.path.split(args.folder)
    csv_out_file = os.path.join(head, "out.csv")
    f = open(csv_out_file, 'w')
    f.write(
        "quantizer,# centroids or bits,q_dim_row,q_dim_col,row_bucketer,num_row_buckets,col_bucketer,num_col_buckets,num_bytes,"
    )

    flat_tasks = []
    for task in config["tasks"]["ws"]:
        flat_tasks.append(task)
    for task in config["tasks"]["analogy"]:
        flat_tasks.append(task + "_add")
        flat_tasks.append(task + "_mul")

    f.write(",".join(flat_tasks) + "," + "\n")

    f.write("baseline,baseline,baseline,baseline,baseline,baseline,baseline,")
    for task in flat_tasks:
        f.write(str(states["baseline"][task]) + ",")
    f.write("\n")

    # Use a regex to extract information from the embedding filename.
    for filename in states:
        if filename == "baseline":
            continue
        matchObj = re.match(
            r'q([^0-9]+)(\d+)b_d(\d+)_(\d+)_r([^0-9]+)(\d+)_c([^0-9]+)(\d+)_bytes(.*).txt',
            filename, re.M | re.I)
        quantizer = matchObj.group(1)
        quantizer_config = matchObj.group(2)
        q_num_r = matchObj.group(3)
        q_num_c = matchObj.group(4)
        row_bucketer = matchObj.group(5)
        num_row_buckets = matchObj.group(6)
        col_bucketer = matchObj.group(7)
        num_col_buckets = matchObj.group(8)
        num_bytes = matchObj.group(9)

        f.write(",".join([quantizer, quantizer_config, q_num_r,\
                q_num_c, row_bucketer, \
                num_row_buckets, col_bucketer, num_col_buckets, num_bytes]))
        f.write(",")
        for task in flat_tasks:
            f.write(str(states[filename][task]) + ",")
        f.write("\n")

    f.close()
    """----------------End write the output to a CSV.------------------------"""

    os.chdir(cwd)

def main():
    args = parse_user_args()
    if args.gen:
        if args.config is None:
            raise ValueError("Configuration file ('-c') must be specified "
                             "by user for embedding generation.")
        config = read_config(args.config)
        gen_embedddings(config, args)
    if args.eval:
        download_intrinsics()
        if args.folder is None:
            raise ValueError(
                "Folder holding (only) the embeddings to be evaluated"
                "must be specified by user for embedding generation.")
        if args.config:
            # Override config.
            config = read_config(args.config)
        else:
            # Default to snapshotted config from generation.
            head, tail = os.path.split(args.folder)
            config = read_config(os.path.join(head, "config.json"))
        eval_embeddings(config, args)


if __name__ == "__main__": main()
