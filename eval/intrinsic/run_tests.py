import argparse
import os
import subprocess
import datetime
import re

################################################################################
num_cores = 64
filenames = [
    "/dfs/scratch0/caberger/systems/GloVe-1.2/vectors.txt"
]

quantizers = ["kmeans", "uniform_fp"]
bucketers = ["kmeans", "uniform"]

kmeans_params = {
    "num_row_buckets": [1, -1],
    "num_col_buckets": [1, -1],
    "sweep": ("num_centroids", [2, 8, 16, 128])
}

uniform_params = {
    "num_row_buckets": [1, -1],
    "num_col_buckets": [1, -1],
    "sweep": ("num_bits", [1, 8])
}
################################################################################

################################################################################
tasks = {
    "ws": [
        "bruni_men",
        "luong_rare",
        "radinsky_mturk",
        "simlex999",
        "ws353_relatedness",
        "ws353_similarity",
        "ws353"
    ],
    "analogy" : [
        "google",
        "msr"
    ]
}

################################################################################



def gen_embedddings():
    cwd = os.getcwd()
    now = datetime.datetime.now()
    output_folder = os.path.join(cwd, "outputs") + "m" + str(now.month) + "-d" \
                    + str(now.day) + "-t" + str(now.hour) + "-m" + str(now.minute)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    logdir = os.path.join(output_folder, "log")
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    os.chdir("../..")
    processes = []
    for filename in filenames:
        for q in quantizers:
            for cb in bucketers:
                for rb in bucketers:
                    params = uniform_params if q == "uniform" else kmeans_params

                    num_row_buckets = params["num_row_buckets"]
                    num_col_buckets = params["num_col_buckets"]

                    for r in num_row_buckets:
                        for c in num_col_buckets:
                            sweep = params["sweep"][0]
                            for s in params["sweep"][1]:
                                command = "python main.py" + \
                                      f" -f {filename}" + \
                                      f" --num_row_buckets {r}" + \
                                      f" --num_col_buckets {c}" + \
                                      f" --{sweep} {s}" + \
                                      f" --quantizer {q}" + \
                                      f" --row_bucketer {rb}" + \
                                      f" --output_folder {output_folder}" + \
                                      f" --col_bucketer {cb}"
                                log_file = os.path.join(logdir, 
                                    f"q{q}rb{rb}cb{cb}{sweep}{s}nr{r}nc{c}.log")
                                command = command + f" 2>&1 | tee {log_file}"
                                proc = subprocess.Popen(command, shell=True)
                                processes.append(proc)

                            if len(processes) >= num_cores:
                                found = False
                                delete_indexes = []
                                while not found:
                                    for i in range(len(processes)):
                                        poll = processes[i].poll()
                                        if poll != None:
                                            found = True
                                            delete_indexes.append(i)
                                for i in range(len(delete_indexes)):
                                    del processes[delete_indexes[i]-i]

    os.chdir(cwd)
    for proc in processes:
        proc.wait()

def eval_embeddings(folder):
    # read all the files in the folder
    states = {}
    processes = []
    
    # Add a for loop for each task. 
    # 
    for filename in os.listdir(folder):
        for task_class in tasks:
            for task in tasks[task_class]:
                if task_class == "ws":
                    cmd = f"python ws_eval.py GLOVE {os.path.join(folder, filename)} testsets/{task_class}/{task}.txt"
                    print(cmd)
                elif task_class == "analogy":
                    cmd = f"python analogy_eval.py GLOVE {os.path.join(folder, filename)} testsets/{task_class}/{task}.txt"
                    print(cmd)
                proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, close_fds=True)
                processes.append((proc, task_class, task, filename, cmd))

                states[filename] = {}

                if len(processes) >= num_cores:
                    found = False
                    while not found:
                        delete_indexes = []
                        for i in range(len(processes)):
                            iproc = processes[i][0]
                            itask_class = processes[i][1]
                            itask = processes[i][2]
                            ifilename = processes[i][3]
                            icmd = processes[i][4]
                            poll = iproc.poll()
                            if poll != None:
                                found = True
                                output = iproc.stdout.read().strip().decode("utf-8").split(" ")
                                if itask_class == "ws":
                                    states[ifilename][itask] = float(output[2])
                                elif itask_class == "analogy":
                                    states[ifilename][itask + "_add"] = float(output[2])
                                    states[ifilename][itask + "_mul"] = float(output[3])

                                delete_indexes.append(i)
                    for i in range(len(delete_indexes)):
                        del processes[delete_indexes[i]-i]

    for i in range(len(processes)):
        proc = processes[i][0]
        task_class = processes[i][1]
        task = processes[i][2]
        filename = processes[i][3]
        cmd = processes[i][4]

        proc.wait()
        output = proc.stdout.read().strip().decode("utf-8").split(" ")
        if task_class == "ws":
            states[filename][task] = float(output[2])
        elif task_class == "analogy":
            states[filename][task + "_add"] = float(output[2])
            states[filename][task + "_mul"] = float(output[3])


    head, tail = os.path.split(folder)
    csv_out_file = os.path.join(head, "out.csv")
    f = open(csv_out_file, 'w')
    f.write("quantizer")
    f.write(",")
    f.write("# centroids or bits")
    f.write(",")
    f.write("row_bucketer")
    f.write(",")
    f.write("num_row_buckets")
    f.write(",")
    f.write("col_bucketer")
    f.write(",")
    f.write("num_col_buckets")
    f.write(",")
    f.write("num_bytes")
    f.write(",")

    flat_tasks = [ 
        "bruni_men",
        "luong_rare",
        "radinsky_mturk",
        "simlex999",
        "ws353_relatedness",
        "ws353_similarity",
        "ws353",
        "google_mul",
        "google_add",
        "msr_mul",
        "msr_add"
    ]
    f.write(",".join(flat_tasks) + "," + "\n")

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
    # parse the file names

parser = argparse.ArgumentParser()
parser.add_argument(
    "-g", "--gen", action="store_true", help="Flag to generate embeddings.")
parser.add_argument(
    "-e", "--eval", action="store_true", help="Flag to evaluate embeddings.")
parser.add_argument(
    "-f", "--folder", action="store", help="Folder holding the emebeddings to be evaluated.")
args = parser.parse_args()

if args.gen:
    gen_embedddings()
elif args.eval:
    eval_embeddings(args.folder)
