#!/usr/bin/env bash
salloc -n 1 -N 1 -t 0:15:00 -p notchpeak-shared-short -A notchpeak-shared-short --gres=gpu:k80:1
# salloc -n 16 -t 0:15:00 -p notchpeak-shared-short -A notchpeak-shared-short

