docker run -it  \
-u $(id -u):$(id -g) \
--name my_try8 \
-v /media/data:/workspace/data_local \
-v /media/MedIP-Praxisprojekt:/workspace/MedIP-PP \
-v /home/WIN-UNI-DUE/smbohann:/workspace/smbohann \
--gpus all \
--shm-size 2G \
smbohann/bvm116:latest
 