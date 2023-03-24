#!/bin/bash
output="merged.mp4"
for file in voxel_video/car/*.mp4; do
  if [ -f "$file" ]; then
    if [ "$file" != "$output" ]; then
      if [ ! -f "$output" ]; then
        ffmpeg -i "$file" -c copy "$output"
      else
        ffmpeg -i "$output" -i "$file" -filter_complex "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1" -c:v libx264 -crf 18 -preset veryfast -c:a aac -b:a 192k "$output.tmp.mp4"
        mv "$output.tmp.mp4" "$output"
      fi
    fi
  fi
done

