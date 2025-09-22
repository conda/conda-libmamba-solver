FROM continuumio/miniconda3:latest

RUN conda install --yes psutil
COPY profile.py /root/

CMD ["python", "/root/profile.py"]
