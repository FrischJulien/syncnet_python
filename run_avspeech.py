#make sure the file exists
!sudo touch /mnt/efs/fs1/avspeech/output.txt
!sudo chmod 777 /mnt/efs/fs1/avspeech/output.txt

#run the avspeech script
data_path = "/mnt/efs/fs1/avspeech/raw/xad/"
tmp_dir="./temp/"
syncnet_model="./data/syncnet_v2.model"
min_track=10
min_face_size = 288
batch_size=32
sample_duration=1
parts_num=8
output_path="/mnt/efs/fs1/avspeech/output.txt"

subcommande = ""
for p in range(parts_num):
    part=p
    subcommande+="CUDA_VISIBLE_DEVICES={} python analyze_full_paralle2.py --data_path {} --tmp_dir {} --syncnet_model {} --min_track {} --min_face_size {} --batch_size {} --sample_duration {} --parts_num {} --part {} --output_path {}".format(part,data_path,tmp_dir,syncnet_model,min_track,min_face_size,batch_size,sample_duration,parts_num,part,output_path)
    if p<parts_num-1:
        subcommande+=" & "
print(subcommande)
        
#!python analyze_full_paralle2.py --data_path $data_path --tmp_dir $tmp_dir --syncnet_model $syncnet_model --min_track $min_track --min_face_size $min_face_size --batch_size $batch_size --ngpu $ngpu --sample_duration $sample_duration --parts_num $parts_num --part $part

import subprocess
subprocess.run(subcommande, shell=True)