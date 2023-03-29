#!/bin/bash
# convert folder of output frames to mp4 video output.mp4 in the same folder
ffmpeg -framerate 10 -f image2  -i %04d.png -c:v libx264 -crf 23 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" output.mp4