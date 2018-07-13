import argparse
import os
import subprocess


filnames = [
    "/Users/christopheraberger/Research/code/GloVe-1.2/vectors.txt"
]

quantizers = ["kmeans", "uniform_fp"]
bucketers = ["kmeans", "uniform"]

kmeans_params = {
    "num_row_buckets": [1, 100],
    "num_col_buckets": [1, 100],
    "sweep": ("num_centroids", [2, 8, 16, 128])
}

uniform_params = {
    "num_row_buckets": [1, 100],
    "num_col_buckets": [1, 100],
    "sweep": ("num_bits", [1, 8])
}


def gen_embedddings():
    cwd = os.getcwd()
    if not os.path.exists("log"):
        os.makedirs("log")
    logdir = os.path.join(cwd, "log")

    os.chdir("../..")
    processes = []
    for filename in filenames:
        for q in quantizers:
            for b in bucketers:
                params = uniform_params if b == "uniform" else kmeans_params

                num_row_buckets = params["num_row_buckets"]
                num_col_buckets = params["num_col_buckets"]

                for r in num_row_buckets:
                    for c in num_col_buckets:
                        sweep = params["sweep"][0]
                        for s in params["sweep"][1]:
                            command = f"python main.py" + \
                                  f" -f {filename}" + \
                                  f" --num_row_buckets {r}" + \
                                  f" --num_col_buckets {c}" + \
                                  f" --{sweep} {s}" + \
                                  f" --quantizer {q}" + \
                                  f" --bucketer {b}"
                            log_file = os.path.join(logdir, 
                                f"q{q}b{b}{sweep}{s}nr{r}nc{c}.log")
                            command = command + f" 2>&1 | tee {log_file}"
                            proc = subprocess.Popen(command, shell=True)
                            processes.append(proc)

    os.chdir(cwd)
    for proc in processes:
        proc.wait()

parser = argparse.ArgumentParser()
parser.add_argument(
    "-g", "--gen", action="store_true", help="Flag to generate embeddings.")
parser.add_argument(
    "-e", "--eval", action="store_true", help="Flag to evaluate embeddings.")
args = parser.parse_args()

if args.gen:
    gen_embedddings()
