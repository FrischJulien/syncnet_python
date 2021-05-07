#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, pickle, os, gzip, glob, shutil
import tqdm

from SyncNetInstance import *

from multiprocessing import Pool
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)




# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser(description = "SyncNet");

parser.add_argument('--initial_model', type=str, required=True);
parser.add_argument('--input_dir', type=str, required=True);
parser.add_argument('--output_dir', type=str,required=True);
parser.add_argument('--ngpu', type=int,required=True);

#default arguments, did not look into it
parser.add_argument('--data_dir', type=str, default='data/work');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');

opt = parser.parse_args();



# ==================== LOAD MODEL AND FILE LIST ====================

s = SyncNetInstance();
s.loadParameters(opt.initial_model);
#print("Model %s loaded."%opt.initial_model);
flist = glob.glob(os.path.join(opt.input_dir,'*/*.mp4'))
flist.sort()

#pbar = tqdm(total=len(flist))


def processfile(inputname):
    setattr(opt,'reference',os.path.basename(inputname)[:-4])
    outputname = os.path.join(opt.output_dir,os.path.basename(inputname))
    offset = s.evaluate(opt,videofile=inputname)


# ==================== GET OFFSETS ====================

if __name__ == '__main__':
    mp.freeze_support()
    #with Pool(opt.ngpu) as p:
    #    p.map(processfile,flist)
    #    p.close()
    #    p.join()
    p=Pool(opt.ngpu)
    for _ in tqdm.tqdm(p.imap_unordered(processfile,flist), total=len(flist)):
        pass
    p.close()
    p.join
    

 


        
