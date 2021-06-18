import time, pdb, argparse, subprocess, pickle, os, gzip, glob
import torch
import numpy as np
import time, pdb, argparse, subprocess, os, math, glob
import cv2
import python_speech_features
from tqdm import tqdm
import timeit

from scipy import signal
from scipy.io import wavfile
from SyncNetModel import *
from shutil import rmtree

import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector


import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal

from detectors import S3FD


# ========== ========== ========== ==========
# # PARSE ARGS
# ========== ========== ========== ==========

parser = argparse.ArgumentParser(description = "SyncNet");
parser.add_argument('--syncnet_model', type=str, required=True);
parser.add_argument('--batch_size', type=int, required=True, help='');
parser.add_argument('--vshift', type=int, default='15', help='');
#parser.add_argument('--data_dir', type=str, default='data/work', help='');
#parser.add_argument('--videofile', type=str, required=True);
parser.add_argument('--data_path', type=str, required=True);
parser.add_argument('--tmp_dir', type=str, required=True);
parser.add_argument('--facedet_scale',  type=float, default=0.25, help='Scale factor for face detection');
parser.add_argument('--min_track', type=int, required=True,  help='Minimum facetrack duration');
parser.add_argument('--num_failed_det', type=int, default=25,   help='Number of missed detections allowed before tracking is stopped');
parser.add_argument('--min_face_size',  type=int, required=True,  help='Minimum face size in pixels');
parser.add_argument('--frame_rate',     type=int, default=25,   help='Frame rate');
parser.add_argument('--crop_scale',     type=float, default=0.40, help='Scale bounding box');
parser.add_argument('--verbose',    default=False, help='');
parser.add_argument('--nthreads', type=int,default=1);
parser.add_argument('--sample_duration',type=float, required=True)
parser.add_argument('--parts_num', type=int, required=True)
parser.add_argument('--part', type=int, required=True)
parser.add_argument('--output_path', type=str, required=True);



opt = parser.parse_args();

setattr(opt,'avi_dir',os.path.join(opt.tmp_dir,'pyavi'))
#setattr(opt,'pytmp_dir',os.path.join(opt.tmp_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.tmp_dir,'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.tmp_dir,'pycrop'))
setattr(opt,'frames_dir',os.path.join(opt.tmp_dir,'pyframes'))

# ========== ========== ========== ==========
# # SyncnetInstanceAnalyze Class
# ========== ========== ========== ==========

class SyncNetInstanceAnalyze(torch.nn.Module):

    def __init__(self, dropout = 0, num_layers_in_fc_layers = 1024):
        super(SyncNetInstanceAnalyze, self).__init__();

        self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers).cuda();

    def evaluate(self, opt,reference, videofile):

        self.__S__.eval();

        # ==================== Convert Files ====================

        if os.path.exists(os.path.join(opt.tmp_dir,reference)):
          rmtree(os.path.join(opt.tmp_dir,reference))

        os.makedirs(os.path.join(opt.tmp_dir,reference))

        command = ("ffmpeg -loglevel error -y -i %s -threads 1 -f image2 %s" % (videofile,os.path.join(opt.tmp_dir,reference,'%06d.jpg'))) 
        output = subprocess.call(command, shell=True, stdout=None)
        
        #command = ("ffmpeg -loglevel error -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (videofile,os.path.join(opt.tmp_dir,opt.reference,'audio.wav')))             
        #output = subprocess.call(command, shell=True, stdout=None)
        
        # ==================== Load Videos ====================

        images = []
        
        flist = glob.glob(os.path.join(opt.tmp_dir,reference,'*.jpg'))
        flist.sort()

        for fname in flist:
            images.append(cv2.imread(fname))

        im = np.stack(images,axis=3)
        im = np.expand_dims(im,axis=0)
        im = np.transpose(im,(0,3,4,1,2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        # ==================== Load Audios ====================

        sample_rate, audio = wavfile.read(os.path.join(opt.avi_dir,reference,'audio.wav'))
        mfcc = zip(*python_speech_features.mfcc(audio,sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])

        cc = np.expand_dims(np.expand_dims(mfcc,axis=0),axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

        # ==================== Check Audio and Video input length ====================

        #if (float(len(audio))/16000) != (float(len(images))/25) :
            #print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different."%(float(len(audio))/16000,float(len(images))/25))

        min_length = min(len(images),math.floor(len(audio)/640))
        
        # ==================== Generate Video and Audio feats ====================

        lastframe = min_length-5
        im_feat = []
        cc_feat = []

        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):
            
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.__S__.forward_lip(im_in.cuda());
            im_feat.append(im_out.data.cpu())

            cc_batch = [ cct[:,:,:,vframe*4:vframe*4+20] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            cc_in = torch.cat(cc_batch,0)
            cc_out  = self.__S__.forward_aud(cc_in.cuda())
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat,0)
        cc_feat = torch.cat(cc_feat,0)

        # ==================== Compute Offset ====================
            
        #print('Compute time %.3f sec.' % (time.time()-tS))

        dists = calc_pdist(im_feat,cc_feat,vshift=opt.vshift)
        mdist = torch.mean(torch.stack(dists,1),1)

        minval, minidx = torch.min(mdist,0)

        offset = opt.vshift-minidx
        conf   = torch.median(mdist) - minval

        fdist   = np.stack([dist[minidx].numpy() for dist in dists])
        # fdist   = numpy.pad(fdist, (3,3), 'constant', constant_values=15)
        fconf   = torch.median(mdist).numpy() - fdist
        fconfm  = signal.medfilt(fconf,kernel_size=9)
        
        #np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        #print('Framewise conf: ')
        #print(fconfm)
        #print('AV offset: \t%d \nMin dist: \t%.3f\nConfidence: \t%.3f' % (offset,minval,conf))

        dists_npy = np.array([ dist.numpy() for dist in dists ])
        #return offset.numpy(), conf.numpy(), dists_npy
        return offset.numpy(), conf.numpy(), minval.numpy()

    def extract_feature(self, opt, videofile):

        self.__S__.eval();
        
        # ==================== Load videos ====================
        cap = cv2.VideoCapture(videofile)

        frame_num = 1;
        images = []
        while frame_num:
            frame_num += 1
            ret, image = cap.read()
            if ret == 0:
                break

            images.append(image)

        im = np.stack(images,axis=3)
        im = np.expand_dims(im,axis=0)
        im = np.transpose(im,(0,3,4,1,2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        
        # ==================== Generate Video Feats ====================

        lastframe = len(images)-4
        im_feat = []

        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):
            
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.__S__.forward_lipfeat(im_in.cuda());
            im_feat.append(im_out.data.cpu())

        im_feat = torch.cat(im_feat,0)

            # ==================== Return video feat ====================
            
        #print('Compute time %.3f sec.' % (time.time()-tS))

        return im_feat


    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage);

        self_state = self.__S__.state_dict();

        for name, param in loaded_state.items():

            self_state[name].copy_(param);
                       
# ========== ========== ========== ==========
# # FACE DETECTION
# ========== ========== ========== ==========

def inference_video(opt,reference,DET):

  flist = glob.glob(os.path.join(opt.frames_dir,reference,'*.jpg'))
  flist.sort()

  dets = []
      
  for fidx, fname in enumerate(flist):

    start_time = time.time()
    
    image = cv2.imread(fname)

    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])

    dets.append([]);
    for bbox in bboxes:
      dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})

    elapsed_time = time.time() - start_time

    #dprint('%s-%05d; %d dets; %.2f Hz' % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),fidx,len(dets[-1]),(1/elapsed_time))) 

  savepath = os.path.join(opt.work_dir,reference,'faces.pckl')

  with open(savepath, 'wb') as fil:
    pickle.dump(dets, fil)

  return dets


# ========== ========== ========== ==========
# # SCENE DETECTION
# ========== ========== ========== ==========

def scene_detect(opt, reference):
  video_manager = VideoManager([os.path.join(opt.avi_dir,reference,'video.avi')])
  stats_manager = StatsManager()
  scene_manager = SceneManager(stats_manager)
  # Add ContentDetector algorithm (constructor takes detector options like threshold).
  scene_manager.add_detector(ContentDetector())
  base_timecode = video_manager.get_base_timecode()

  video_manager.set_downscale_factor()

  video_manager.start()

  scene_manager.detect_scenes(frame_source=video_manager,show_progress=False)

  scene_list = scene_manager.get_scene_list(base_timecode)

  savepath = os.path.join(opt.work_dir,reference,'scene.pckl')

  if scene_list == []:
    scene_list = [(video_manager.get_base_timecode(),video_manager.get_current_timecode())]

  with open(savepath, 'wb') as fil:
    pickle.dump(scene_list, fil)
  #print('%s - scenes detected %d'%(os.path.join(opt.avi_dir,opt.reference,'video.avi'),len(scene_list)))

  return scene_list

# ========== ========== ========== ==========
# # FACE TRACKING
# ========== ========== ========== ==========

def track_shot(opt,scenefaces):

  iouThres  = 0.5     # Minimum IOU between consecutive face detections
  tracks    = []
  face_sizes = []

  while True:
    track     = []
    for framefaces in scenefaces:
      for face in framefaces:
        if track == []:
          track.append(face)
          framefaces.remove(face)
        elif face['frame'] - track[-1]['frame'] <= opt.num_failed_det:
          iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
          if iou > iouThres:
            track.append(face)
            framefaces.remove(face)
            continue
        else:
          break

    if track == []:
      break
    elif len(track) > opt.min_track:
      
      framenum    = np.array([ f['frame'] for f in track ])
      bboxes      = np.array([np.array(f['bbox']) for f in track])

      frame_i   = np.arange(framenum[0],framenum[-1]+1)

      bboxes_i    = []
      for ij in range(0,4):
        interpfn  = interp1d(framenum, bboxes[:,ij])
        bboxes_i.append(interpfn(frame_i))
      bboxes_i  = np.stack(bboxes_i, axis=1)

      if max(np.mean(bboxes_i[:,2]-bboxes_i[:,0]), np.mean(bboxes_i[:,3]-bboxes_i[:,1])) > opt.min_face_size:
        tracks.append({'frame':frame_i,'bbox':bboxes_i})
        face_sizes.append(max(np.mean(bboxes_i[:,2]-bboxes_i[:,0]), np.mean(bboxes_i[:,3]-bboxes_i[:,1])))
        #print("face size : {}".format(max(np.mean(bboxes_i[:,2]-bboxes_i[:,0]), np.mean(bboxes_i[:,3]-bboxes_i[:,1]))))
        
  return tracks, face_sizes

# ========== ========== ========== ==========
# # IOU FUNCTION
# ========== ========== ========== ==========

def bb_intersection_over_union(boxA, boxB):
  
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
 
  interArea = max(0, xB - xA) * max(0, yB - yA)
 
  boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
 
  iou = interArea / float(boxAArea + boxBArea - interArea)
 
  return iou

# ========== ========== ========== ==========
# # VIDEO CROP AND SAVE
# ========== ========== ========== ==========
        
def crop_video(opt,reference,track,cropfile):

  flist = glob.glob(os.path.join(opt.frames_dir,reference,'*.jpg'))
  flist.sort()

  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  vOut = cv2.VideoWriter(cropfile+'t.avi', fourcc, opt.frame_rate, (224,224))

  dets = {'x':[], 'y':[], 's':[]}

  for det in track['bbox']:

    dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2) 
    dets['y'].append((det[1]+det[3])/2) # crop center x 
    dets['x'].append((det[0]+det[2])/2) # crop center y

  # Smooth detections
  dets['s'] = signal.medfilt(dets['s'],kernel_size=13)   
  dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
  dets['y'] = signal.medfilt(dets['y'],kernel_size=13)

  for fidx, frame in enumerate(track['frame']):

    cs  = opt.crop_scale

    bs  = dets['s'][fidx]   # Detection box size
    bsi = int(bs*(1+2*cs))  # Pad videos by this amount 

    image = cv2.imread(flist[frame])
   
    frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
    my  = dets['y'][fidx]+bsi  # BBox center Y
    mx  = dets['x'][fidx]+bsi  # BBox center X

    face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
    
    vOut.write(cv2.resize(face,(224,224)))

  audiotmp    = os.path.join(opt.tmp_dir,reference,'audio.wav')
  audiostart  = (track['frame'][0])/opt.frame_rate
  audioend    = (track['frame'][-1]+1)/opt.frame_rate

  vOut.release()

# ========== ========== ========== ==========
# # Get OFFSET
# ========== ========== ========== ==========    
def calc_pdist(feat1, feat2, vshift=10):
    
    win_size = vshift*2+1

    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))

    dists = []

    for i in range(0,len(feat1)):

        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))

    return dists


# ========== ========== ========== ==========
# # Get begining and end time for clip sample
# ========== ========== ========== ==========    
def calc_beginingandend(videofile_path,requested_sample_duration):
    
    command=("ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {}".format(videofile_path))
    video_duration = float((subprocess.check_output(command, shell=True)).decode('utf-8').strip())
    sample_duration=min(video_duration,requested_sample_duration)
    begining=video_duration/2-sample_duration/2
    end=video_duration/2+sample_duration/2
    #print("begining: {} ; end : {} ".format(begining,end))

    return begining,end

# ========== ========== ========== ==========
# # MP HANDLER
# ========== ========== ========== ==========

def mp_handler(job):
    try:    
        videofile, opt, gpu_id = job
        if opt.verbose :
            start = timeit.default_timer()
        dir_path=os.path.dirname(videofile)
        videofile = os.path.basename(videofile)
        reference=videofile[:-4]



        # ========== DELETE EXISTING DIRECTORIES ==========

        if os.path.exists(os.path.join(opt.work_dir,reference)):
          rmtree(os.path.join(opt.work_dir,reference))

        if os.path.exists(os.path.join(opt.crop_dir,reference)):
          rmtree(os.path.join(opt.crop_dir,reference))

        if os.path.exists(os.path.join(opt.avi_dir,reference)):
          rmtree(os.path.join(opt.avi_dir,reference))

        if os.path.exists(os.path.join(opt.frames_dir,reference)):
          rmtree(os.path.join(opt.frames_dir,reference))

        if os.path.exists(os.path.join(opt.tmp_dir,reference)):
          rmtree(os.path.join(opt.tmp_dir,reference))

        # ========== MAKE NEW DIRECTORIES ==========

        os.makedirs(os.path.join(opt.work_dir,reference))
        os.makedirs(os.path.join(opt.crop_dir,reference))
        os.makedirs(os.path.join(opt.avi_dir,reference))
        os.makedirs(os.path.join(opt.frames_dir,reference))
        os.makedirs(os.path.join(opt.tmp_dir,reference))
        if opt.verbose : 
            print('Time to delete and make directories: ', timeit.default_timer() - start)  
            start = timeit.default_timer()

        # ========== CONVERT VIDEO AND EXTRACT FRAMES ==========
        begining,end = calc_beginingandend(os.path.join(dir_path,videofile),opt.sample_duration)

        command = ("ffmpeg -loglevel error -y -ss %f -i %s -t %f -qscale:v 2 -async 1 -r 25 %s" % (begining,os.path.join(dir_path,videofile),end-begining,os.path.join(opt.avi_dir,reference,'video.avi')))
    
        #command = ("ffmpeg -loglevel error -y -i %s -qscale:v 2 -async 1 -r 25 %s" % (os.path.join(dir_path,videofile),os.path.join(opt.avi_dir,reference,'video.avi')))
        output = subprocess.call(command, shell=True, stdout=None)

        command = ("ffmpeg -loglevel error -y -i %s -qscale:v 2 -threads 1 -f image2 %s" % (os.path.join(opt.avi_dir,reference,'video.avi'),os.path.join(opt.frames_dir,reference,'%06d.jpg'))) 
        output = subprocess.call(command, shell=True, stdout=None)

        command = ("ffmpeg -loglevel error -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (os.path.join(opt.avi_dir,reference,'video.avi'),os.path.join(opt.avi_dir,reference,'audio.wav'))) 
        output = subprocess.call(command, shell=True, stdout=None)

        if opt.verbose :
            print('Convert video and extract frames: ', timeit.default_timer() - start)  
            start = timeit.default_timer()
    
        # ========== FACE DETECTION ==========
        faces = inference_video(opt,reference,DET)
        if opt.verbose : 
            print('Face Detection: ', timeit.default_timer() - start)  
            start = timeit.default_timer()
    
        # ========== SCENE DETECTION ==========
        scene = scene_detect(opt,reference)
        if opt.verbose:
            print('Scene Detection: ', timeit.default_timer() - start)  
            start = timeit.default_timer()
    
        # ========== FACE TRACKING ==========
        alltracks = []
        vidtracks = []
        face_sizes = []

        for shot in scene:

          if shot[1].frame_num - shot[0].frame_num >= opt.min_track :
            new_track, face_size=track_shot(opt,faces[shot[0].frame_num:shot[1].frame_num])
            #alltracks.extend(track_shot(opt,faces[shot[0].frame_num:shot[1].frame_num]))
            if len(face_size)>0:
                alltracks.extend(new_track)
                face_sizes.append(face_size)
        if opt.verbose:
            print('Face Tracking: ', timeit.default_timer() - start)  
            start = timeit.default_timer()
    
        # ========== FACE TRACK CROP ==========

        for ii, track in enumerate(alltracks):
            vidtracks.append(crop_video(opt,reference,track,os.path.join(opt.crop_dir,reference,'%05d'%ii)))
        if opt.verbose:
            start = timeit.default_timer()
        # ========== SAVE RESULTS ==========

        savepath = os.path.join(opt.work_dir,reference,'tracks.pckl')

        with open(savepath, 'wb') as fil:
            pickle.dump(vidtracks, fil)
        rmtree(os.path.join(opt.tmp_dir,reference))
    
        if opt.verbose:
            print('Save Results: ', timeit.default_timer() - start)  
            start = timeit.default_timer()
        # ==================== LOAD FILE LIST ====================

        flist = glob.glob(os.path.join(opt.crop_dir,reference,'0*.avi'))
        flist.sort()
        if opt.verbose:
            print('Load File List: ', timeit.default_timer() - start)  
            start = timeit.default_timer()
        # ==================== GET OFFSETS ====================

        dists = []
    #for idx, fname in enumerate(flist):
    #    print("idx: {}".format(idx))
    #    offset, conf, dist = s.evaluate(opt,reference, videofile=fname)
    #    dists.append(dist)
    #    with open("output.txt", "a") as file_object:
    #        file_object.write("filename : {} ; offset : {} ; conf : {} ; mdist : {}; face_size : {}".format(videofile,offset,conf,dist,face_sizes[0][idx]))
    #        file_object.write("\n")
        if len(flist)==1 :
            fname = flist[0]
            offset, conf, dist = s.evaluate(opt,reference, videofile=fname)
            dists.append(dist)
            with open(opt.output_path, "a") as file_object:
                file_object.write("part : {} ; filename : {} ; offset : {} ; conf : {} ; mdist : {}; face_size : {}\n".format(opt.part,(dir_path+"/")+videofile,offset,conf,dist,face_sizes[0][0]))
        if opt.verbose:
            print('Get & print offset : ', timeit.default_timer() - start)  
    
   # ========== DELETE USED DIRECTORIES ==========
        if os.path.exists(os.path.join(opt.work_dir,reference)):
          rmtree(os.path.join(opt.work_dir,reference))

        if os.path.exists(os.path.join(opt.crop_dir,reference)):
          rmtree(os.path.join(opt.crop_dir,reference))

        if os.path.exists(os.path.join(opt.avi_dir,reference)):
          rmtree(os.path.join(opt.avi_dir,reference))

        if os.path.exists(os.path.join(opt.frames_dir,reference)):
          rmtree(os.path.join(opt.frames_dir,reference))
    except:
        pass

# ========== ========== ========== ==========
# # MAIN
# ========== ========== ========== ==========

# ========== LOADING MANDATORY STUFF ==========
DET = S3FD(device='cuda')
s = SyncNetInstanceAnalyze();
s.loadParameters(opt.syncnet_model);
#print("Model %s loaded."%opt.syncnet_model);

# ========== FOR LOOP OVER ALL EXISTING MP4 FILES IN DATA_PATH ==========

videofiles = glob.glob(opt.data_path+'*/*.mp4')
videofiles = [v for i, v in enumerate(videofiles) if (i%opt.parts_num)==opt.part]


#print('Started processing for {} GPUs'.format(opt.nthreads))

jobs = [(vfile, opt, i%opt.nthreads) for i, vfile in enumerate(videofiles)]
if opt.nthreads==1:
    mp_handler(jobs[0])
else:
    p = ThreadPoolExecutor(opt.nthreads)
    futures = [p.submit(mp_handler, j) for j in jobs]
    _ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]


      
# ==================== PRINT RESULTS TO FILE ====================

#with open(os.path.join(opt.work_dir,opt.reference,'activesd.pckl'), 'wb') as fil:
#    pickle.dump(dists, fil)