import sys
import argparse
from utils.conf import Configuration
from utils.expe import Experiment
import time

def main(argv):
    parser = argparse.ArgumentParser(description='Command-line parameters for MySEAnet')
    parser.add_argument('-C', '--conf', type=str, required=True, dest='confpath', help='path of conf file')
    args = parser.parse_args(argv[1: ])
    conf = Configuration(args.confpath)
    expe = Experiment(conf)
    expe.run()

if __name__ == '__main__':
    start_time = time.time()
    main(sys.argv)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"during: {elapsed_time / 60} min")