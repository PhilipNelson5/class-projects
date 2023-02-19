#!/usr/bin/env bash

if [ $# -lt 2 ]
then
  echo "requires two arguments: width height [frames = 300]"
  exit 0
fi

#./build/gif $1 $2 $3 | ffmpeg -y -f rawvideo -pixel_format argb -video_size $1x$2 -i - -c:v libx264 -pix_fmt yuv444p out.mp4
./build/gif $1 $2 $3 | ffmpeg -y -f rawvideo -r 3 -pixel_format argb -video_size $1x$2 -i - -c:v libx264 -b:v 5000k -pix_fmt yuv444p out.mp4
