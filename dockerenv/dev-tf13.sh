#!/bin/sh
xhost local:root
docker run -it \
--runtime=nvidia \
--name tf13 \
--rm --network=host \
-v /media/wildcard/daa/work:/code \
-v /media/wildcard/daa/data:/data \
--volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
--env="DISPLAY=$DISPLAY" \
--device /dev/dri:/dev/dri \
--ipc host \
sfr-tf:1.13.0 /bin/bash