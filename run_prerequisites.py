import os

# EFS

os.system('sudo chmod ugo+rwx /mnt/efs/fs1')

# Requirements
os.system('pip3 install -r requirements.txt')
os.system('conda install ffmpeg -y')
os.system('sh download_model.sh')
