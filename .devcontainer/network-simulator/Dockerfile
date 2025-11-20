FROM continuumio/miniconda3:latest
COPY apt-deps.txt post_create.sh post_start.sh ./
RUN bash ./post_create.sh
CMD [ "bash", "post_start.sh" ]