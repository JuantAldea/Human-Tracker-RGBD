#!/bin/bash

ffmpeg -framerate 18 -i color_%d.png -r 18 -vcodec png  video0_color.avi
ffmpeg -framerate 18 -i depth_1_3_%d.png -r 18 -vcodec png  video0_depth_1_3.avi
ffmpeg -framerate 18 -i depth_4_%d.png -r 18 -vcodec png  video0_depth_4.avi
