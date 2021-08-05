#!/bin/bash

# Build
# docker build -t lungdl .

# Run

# Train         
docker run --rm --gpus all \
           -v /home/matt/Documents/Projects/vnet_lung_tumors/data/:/home/app/data \
           -v /home/matt/Documents/Projects/vnet_lung_tumors/tmp:/home/app/tmp \
           -v /home/matt/Documents/Projects/vnet_lung_tumors/:/home/app \
           -it lungdl python main.py --config_json config_MH_xfer.json
       
# Evaluate    
docker run --rm --gpus 0 \
           -v /home/matt/Documents/Projects/vnet_lung_tumors/data/:/home/app/data \
           -v /home/matt/Documents/Projects/vnet_lung_tumors/tmp:/home/app/tmp \
           -v /home/matt/Documents/Projects/vnet_lung_tumors/:/home/app \
           -it lungdl python main.py -p evaluate --config_json config_MH_training_sims.json
           
