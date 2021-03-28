#!/usr/bin/env bash

mpirun -np 3 -hostfile ./mpi_host_file python3 ./main_fedavg.py \
  --gpu_server_num 1 \
  --gpu_num_per_server 1 \
  --model lr \
  --partition_method hetero \
  --client_num_in_total 10 \
  --client_num_per_round 2 \
  --comm_round 4 \
  --epochs 2 \
  --batch_size 8 \
  --lr 0.003 \
  --ci 1 \
  --sql_host "localhost" \
  --sql_user root \
  --sql_password root \
  --sql_database mnist \
  --influence_test_clients 300
