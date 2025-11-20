FROM continuumio/miniconda3:latest
COPY apt-deps.txt apt_deps.sh conda_deps.sh post_create.sh post_start.sh ./
RUN bash ./apt_deps.sh
# conda_deps.sh wants mounts, which don't exist until the container runs.
CMD [ "bash", "./conda_deps.sh" ]