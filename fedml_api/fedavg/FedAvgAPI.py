import time

from mpi4py import MPI

from fedml_api.fedavg.FedAVGAggregator import FedAVGAggregator
from fedml_api.fedavg.FedAVGTrainer import FedAVGTrainer
from fedml_api.fedavg.FedAvgClientManager import FedAVGClientManager
from fedml_api.fedavg.FedAvgServerManager import FedAVGServerManager


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedAvg_distributed(process_id, worker_number, device, comm, model, args):
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, model)
    else:
        init_client(args, device, comm, process_id, worker_number, model)


def init_server(args, device, comm, rank, size, model):
    # aggregator
    worker_num = size - 1
    aggregator = FedAVGAggregator(worker_num, device, model, args)

    # start the distributed training
    server_manager = FedAVGServerManager(args, aggregator, comm, rank, size)
    server_manager.send_init_msg()
    server_manager.run()


def init_client(args, device, comm, process_id, size, model):
    # trainer
    client_index = process_id - 1
    trainer = FedAVGTrainer(client_index, device, model, args)

    client_manager = FedAVGClientManager(args, trainer, comm, process_id, size)
    client_manager.run()
