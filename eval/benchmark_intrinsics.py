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
        "-g", "--gen", action="store_true", 
        help="Flag to generate embeddings.")
    parser.add_argument(
        "-c", "--config", action="store", 
        help="Configuration file for generating embeddings.")
    parser.add_argument(
        "-e", "--eval", action="store_true", 
        help="Flag to evaluate embeddings.")
    parser.add_argument(
        "-f", "--folder", action="store", 
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

def poll_processes(processes, num_cores, state = None):
    if len(processes) < num_cores:
        return
    found = False
    while not found:
        delete_indexes = []
        for i in range(len(processes)):
            poll = processes[i].poll()
            if poll != None:
                proc.update_state(state)
                found = True
                delete_indexes.append(i)
        for i in range(len(delete_indexes)):
            del processes[delete_indexes[i]-i]

def wait_processes(processes, state = None):
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

    def get_gen_cmd(filename, r, c, sweep, s, q, rb, outdir, cb):
        cmd = f"python main.py" + \
                    f" -f {filename}" + \
                    f" --num_row_buckets {r}" + \
                    f" --num_col_buckets {c}" + \
                    f" --{sweep} {s}" + \
                    f" --quantizer {q}" + \
                    f" --row_bucketer {rb}" + \
                    f" --output_folder {outdir}" + \
                    f" --col_bucketer {cb}"
        log_file = os.path.join(logdir, f"q{q}_r{rb}{r}_c{cb}{c}_{sweep}{s}.log")
        return cmd + f" 2>&1 | tee {log_file}"

    class GenProcess:
        def __init__(self, p):
            self.p = p

        def poll(self):
            self.p.poll()

        def wait(self):
            self.p.wait()

        def update_state(self, state = None):
            return

    outdir, logdir = get_outdir_and_logdir(config, args)

    filename = config["filename"]
    processes = []
    for q in config["quantizers"]:
        for cb in config["bucketers"]:
            for rb in config["bucketers"]:
                sweep = "num_centroids" if q == "kmeans" else "num_bits"
                for r in config["num_row_buckets"]:
                    for c in config["num_col_buckets"]:
                        if is_invalid_config(rb, r, cb, c):
                            break
                        for s in config[sweep]:
                            cmd = get_gen_cmd(filename, r, c, sweep, s, q, 
                                              rb, outdir, cb)
      
                            proc = subprocess.Popen(cmd, shell=True)
                            processes.append(GenProcess(proc))
                            print(cmd)
                            poll_processes(processes, config["num_cores"])

    wait_processes()


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
        def __init__(self, p, task, task_class, filename, cmd):
            self.p = p
            self.task = task
            self.task_class = task_class
            self.filename = filename
            self.cmd = cmd

        def poll(self):
            self.p.poll()

        def wait(self):
            self.p.wait()

        def update_state(self, states):
            output = self.p.stdout.read().strip().decode("utf-8").split(" ")
            if self.task_class == "ws":
                if len(output) != 3:
                    states[self.filename][self.task] = float(0.0)
                else: 
                    states[self.filename][self.task] = float(output[2])
            elif self.task_class == "analogy":
                if len(output) != 4:
                    states[self.filename][self.task + "_add"] = float(0.0)
                    states[self.filename][self.task + "_mul"] = float(0.0)
                else:
                    states[self.filename][self.task + "_add"] = float(output[2])
                    states[self.filename][self.task + "_mul"] = float(output[3])


    # Holds the state.
    states = {}
    processes = []
    
    # Run the baselines first.
    states["baseline"] = {}
    for task_class in tasks:
        for task in tasks[task_class]:
            get_eval_cmd(task, task_class, config["filename"])
            proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, 
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                        close_fds=True)
            processes.append(EvalProcess(proc, task_class, task, 
                                         "baseline", cmd))
            poll_processes(processes, config["num_cores"], states)

    # Now run all the other generated embeddings.
    for filename in os.listdir(args.folder):
        states[filename] = {}
        for task_class in tasks:
            for task in tasks[task_class]:
                get_eval_cmd(task, task_class, os.path.join(args.folder, filename))

                proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, 
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                            close_fds=True)
                processes.append(EvalProcess(proc, task_class, task, 
                                             filename, cmd))

                poll_processes(processes, config["num_cores"], states)

    wait_processes(states)

    """
    Write the output to a CSV.
    """
    head, tail = os.path.split(args.folder)
    csv_out_file = os.path.join(head, "out.csv")
    f = open(csv_out_file, 'w')
    f.write("quantizer,# centroids or bits,row_bucketer,num_row_buckets,col_bucketer,num_col_buckets,num_bytes")

    flat_tasks = []
    for task in tasks["ws"]:
        flat_tasks.append(task)
    for task in tasks["analogy"]:
        flat_tasks.append(task + "_add")
        flat_tasks.append(task + "_mul")

    f.write(",".join(flat_tasks) + "," + "\n")

    f.write("baseline,baseline,baseline,baseline,baseline,baseline,baseline")
    for task in flat_tasks:
        f.write(str(states["baseline"][task]) + ",")
    f.write("\n")

    for filename in states:
        matchObj = re.match( r'q([^0-9]+)(\d+)b_r([^0-9]+)(\d+)_c([^0-9]+)(\d+)_bytes(.*).txt', filename, re.M|re.I)
        quantizer = matchObj.group(1)
        quantizer_config = matchObj.group(2)
        row_bucketer = matchObj.group(3)
        num_row_buckets = matchObj.group(4)
        col_bucketer = matchObj.group(5)
        num_col_buckets = matchObj.group(6)
        num_bytes = matchObj.group(7)

        f.write(",".join([quantizer,  quantizer_config, row_bucketer, \
                num_row_buckets, col_bucketer, num_col_buckets, num_bytes]))
        f.write(",")
        for task in flat_tasks:
            f.write(str(states[filename][task]) + ",")
        f.write("\n")

    f.close()

def main():
    download_intrinsics()
    args = parse_user_args()
    if args.gen:
        if args.config is None:
            raise ValueError("Configuration file ('-c') must be specified "
                             "by user for embedding generation.")
        config = read_config(args.config)
        gen_embedddings(config, args)
    if args.eval:
        if args.folder is None:
            raise ValueError("Folder holding (only) the embeddings to be evaluated"
                             "must be specified by user for embedding generation.")
        config = read_config(os.path.join(folder, "config.json"))
        eval_embeddings(config, args)

if __name__ == "__main__": main()