#!/bin/bash

version=20250312

# Build the docker image
docker build -t abenati/vims:$version .

# Push the docker image on dockerhub
docker push abenati/vims:$version