#!/bin/sh

echo "Building dev environment images"
docker build -t sfr-tf:1.12.0 -f tf12/Dockerfile .
docker build -t sfr-tf:1.13.0 -f tf13/Dockerfile .
