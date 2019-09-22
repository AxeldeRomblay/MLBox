FROM continuumio/miniconda3:4.7.10

# example command to build image, assumes working directory is mlbox root directory
#   docker build -t mlbox_dev .

# example command to run docker container, assumes working directory is mlbox root directory
#   docker run -it --rm -v $PWD:/opt/project --name mlbox_dev mlbox_dev /bin/bash



# create staging area for install mlbox into the docker image
WORKDIR /opt/project
COPY . /opt/project

# install mlbox and its pre-requisites
RUN pip install -r requirements.txt && \
    pip install pytest && \
    pip install -e .

