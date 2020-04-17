from multiprocessing import Pool as ThreadPool
from subprocess import call
from Src.run import main as myfunction
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--base", default=490, help="Base counter for Hyper-param search", type=int)
parser.add_argument("--inc", default=0, help="Increment counter for Hyper-param search", type=int)
parser.add_argument("--hyper", default=2, help="Which Hyper param settings", type=int)
args = parser.parse_args()

sequential = 1
parallel = 1

if __name__== "__main__":
# Sequential processing:
    for seq in range(sequential):

        # Parallel processing
        pool = ThreadPool(parallel)
        my_array = []
        for par in range(parallel):
            my_array.append((True, args.inc*sequential*parallel + seq*parallel + par, args.hyper, args.base))

        # my_array = [(True, args.inc * 2, args.hyper, args.base), (True, args.inc * 2 + 1, args.hyper, args.base)]
        results = pool.starmap(myfunction, my_array)

        # close the pool and wait for the work to finish
        pool.close()
        pool.join()

