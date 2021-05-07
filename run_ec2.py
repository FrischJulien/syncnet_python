import sys
import os
from glob import glob
from os import listdir, path
import argparse

unprocessed_root = '//mnt/efs/fs1/data_unprocessed/voxceleb2/dev/'
preprocessed_root = '//mnt/efs/fs1/data_preprocessed_aligned/main/'
unprocessed_folders = glob(path.join(unprocessed_root, '*'))

parser = argparse.ArgumentParser(description = "run EC2");
parser.add_argument('--begining',       type=int, required=True);
parser.add_argument('--end',      type=int, required=True);
opt = parser.parse_args();

from tqdm import trange, tqdm

for i in trange(opt.begining,opt.end):
    unprocessed_folder=unprocessed_folders[i]
    print(unprocessed_folder)
    basename=os.path.basename(os.path.normpath(unprocessed_folder))
    preprocessed_folder=path.join(preprocessed_root,basename)
    os.system("sudo mkdir "+preprocessed_folder)
    print("folder created")
    os.system("sudo chmod ugo+rwx "+preprocessed_folder)
    print("rights given")
    os.system("python run_syncro.py --initial_model data/syncnet_v2.model  --input_dir "+unprocessed_folder+" --output_dir "+preprocessed_folder+" --ngpu 8 --batch_size 20")