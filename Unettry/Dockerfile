FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
RUN pip install SimpleITK==1.2.4
RUN apt-get update
RUN apt-get install git -y
RUN pip install matplotlib
RUN pip install scikit-image
RUN pip install pandas
RUN pip install numpy
RUN pip install sklearn
RUN pip install requests
RUN pip install tensorboardX
RUN pip install datetime
RUN pip install argparse
RUN pip install torchsummary
RUN pip install opencv-python
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y