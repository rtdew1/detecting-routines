import datetime
from argparse import ArgumentParser
import re

print("loading argparse settings")

parser = ArgumentParser()
parser.add_argument("--warmup", dest="WARMUP", type=int, default=800)
parser.add_argument("--samples", dest="SAMPLES", type=int)
parser.add_argument("--out_dir", dest="OUTPUT_NAME", type=str, default=None)
parser.add_argument("--chains", dest="CHAINS", type=int, default=1)
parser.add_argument("--name", dest="NAME", type=str, default=None)
parser.add_argument("--kernel", dest="COV_TYPE", type=str, default="expon")
parser.add_argument(
    "--hiericepts", dest="HIER_ICEPTS", action="store_true", default=False
)
parser.add_argument(
    "--estloc", dest="EST_ICEPT_LOC", action="store_true", default=False
)
parser.add_argument("--ncust", dest="N_CUST_SAMPLE", type=int, default=2000)
parser.add_argument("--process_results", dest="PROCESS", action="store_true", default=False)

args = parser.parse_args()

if args.OUTPUT_NAME is None:
    replacements = [(" ", "_"), (":", "-"), ("[.]", "-")]
    curtime = str(datetime.datetime.now())
    for old, new in replacements:
        curtime = re.sub(old, new, curtime)
    args.OUTPUT_NAME = f"routines_model_{curtime}"