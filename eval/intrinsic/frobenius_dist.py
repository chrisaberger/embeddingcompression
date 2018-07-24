from docopt import docopt
import numpy as np
import argh



def parse_emb(path):
    lines = list(open(path))
    space_sep_lines = []
    for l in lines:
        space_sep_lines.append(l.split(' '))
    dim = len(space_sep_lines[0])-1
    V = len(space_sep_lines)
    mat = np.zeros([V,dim])
    for i in range(0,len(space_sep_lines)):
        for j in range(1,dim+1):
            #loop indexing is weird to skip word in 0-th index
            mat[i,j-1] = float(space_sep_lines[i][j])

    return mat


def run(base_embs_path, infl_embs_path):
#base embs => baseline uncompressed embeddings
#inflated embs => inflated compressed embeddings

    base_embs = parse_emb(base_embs_path)
    infl_embs = parse_emb(infl_embs_path)

    dist = np.linalg.norm(base_embs - infl_embs)
    print(f"Frob dist 4 {infl_embs_path} is {dist}")


parser = argh.ArghParser()
parser.add_commands([run])

if __name__ == "__main__":
    parser.dispatch()
