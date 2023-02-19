#!/usr/bin/env bash
set -e
[ "$(ls -A _images)" ] && rm _images/*
python gif.py
cd _images
for file in *.ppm; do
    convert $file ${file%.*}.png;
done
cd ..
convert -delay 15 -loop 0 `ls -v _images/*.png` dodeca.gif