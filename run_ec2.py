import sys
import os
from glob import glob
from os import listdir, path
import argparse

unprocessed_root = '//mnt/efs/fs1/data_unprocessed/voxceleb2/dev/'
preprocessed_root = '//mnt/efs/fs1/data_unprocessed_aligned/voxceleb2/dev/'
unprocessed_folders = sorted(glob(path.join(unprocessed_root, '*')))

parser = argparse.ArgumentParser(description = "run EC2");
parser.add_argument('--begining', type=int, required=True);
parser.add_argument('--end', type=int, required=True);
parser.add_argument('--ngpu', type=int, required=True);
parser.add_argument('--batch_size', type=int, required=True);
opt = parser.parse_args();

from tqdm import trange, tqdm

print("for "+ str(opt.begining)+" to "+str(opt.end))
for i in trange(opt.begining,opt.end):
    unprocessed_folder=unprocessed_folders[i]
    print(unprocessed_folder)
    basename=os.path.basename(os.path.normpath(unprocessed_folder))
    preprocessed_folder=path.join(preprocessed_root,basename)
    os.system("sudo mkdir "+preprocessed_folder)
    os.system("sudo chmod ugo+rwx "+preprocessed_folder)
    os.system("python3 run_syncro.py --initial_model data/syncnet_v2.model  --input_dir "+unprocessed_folder+" --output_dir "+preprocessed_folder+" --ngpu "+str(opt.ngpu)+" --batch_size "+str(opt.batch_size))