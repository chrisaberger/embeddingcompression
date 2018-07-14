import argparse
import os
import subprocess
import datetime

################################################################################
num_cores = 56
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
                                for proc in processes:
                                    proc.wait()
                                processes = []

    os.chdir(cwd)
    for proc in processes:
        proc.wait()

def eval_embeddings(folder):
    

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
