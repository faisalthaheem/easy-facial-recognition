#!/bin/sh
docker run -it \
--runtime=nvidia \
--name tf12 \
--rm --network=host \
-v /media/wildcard/daa/work:/code \
-v /media/wildcard/daa/data:/data \
--volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
--env="DISPLAY" \
--device /dev/dri:/dev/dri \
--ipc host \
sfr-tf:1.12.0 /bin/bash