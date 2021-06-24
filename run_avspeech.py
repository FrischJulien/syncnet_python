import os
import subprocess
from tqdm import tqdm


#make sure the file exists
os.system('sudo touch /mnt/efs/fs1/avspeech/output.txt')
os.system('sudo chmod 777 /mnt/efs/fs1/avspeech/output.txt')

#run the avspeech script
data_path = "/mnt/efs/fs1/avspeech/raw/"
repositories=["xap/","xaq/","xar/","xas/"]


tmp_dir="./temp/"
syncnet_model="./data/syncnet_v2.model"
min_track=10
min_face_size = 192
batch_size=32
sample_duration=1
parts_num=8
output_path="/mnt/efs/fs1/avspeech/output.txt"


for rep in tqdm(repositories):
    print("Launching process for repository {}\n".format(rep))  
    subcommande = ""
    for p in range(parts_num):
        part=p
        subcommande+="CUDA_VISIBLE_DEVICES={} python3 analyze_full_paralle2.py --data_path {} --tmp_dir {} --syncnet_model {} --min_track {} --min_face_size {} --batch_size {} --sample_duration {} --parts_num {} --part {} --output_path {}".format(part,data_path+rep,tmp_dir,syncnet_model,min_track,min_face_size,batch_size,sample_duration,parts_num,part,output_path)
        if p<parts_num-1:
            subcommande+=" & "       
    subprocess.run(subcommande, shell=True)