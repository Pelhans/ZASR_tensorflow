#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
    python deploy_potevio/demo_server.py \
    --host_ip 10.3.27.97 \
    --host_port 8086
