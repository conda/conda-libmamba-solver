# Profiling for conda-libmamba-solver

## What's this?

This folder contains profiling scripts for conda-libmamba-solver. These profiling
scripts aim to measure the following parameters:

- CPU Time (User and System)
- Peak Memory usage
- Total network activity (send/receive)
- Peak CPU usage

Ideally, these profiling scripts will be run inside a Docker container so that
network usage is properly isolated, but they can also be run on the host OS
as long as it is macOS or Linux.

## Usage

To run the profile scripts in a Docker container, run the following commands...

To build the image:

```commandline
docker build -t cls-dev-profiler .
```

To run the profiler:

```commandline
docker run --rm cls-dev-profiler
```
