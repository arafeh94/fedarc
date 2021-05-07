import logging
import os
import sys

import numpy
from torch import nn

from src.core import cflog
from src.core.context import Context, LOCAL_COMM

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./../")))
from mpi4py import MPI
import src.api.pipes as pipes
from src.api.sources import ModelReceiver, MnistStreamSource
from src.core.aoi.abs_streamer import Streamer
from fedml_api.model.linear.lr import LogisticRegression

LOCAL_COMM.SIZE = 2

context = Context.Builder('local').build()

mnist_source = MnistStreamSource("localhost", "root", "root", "mnist")
mnist_streamer = Streamer(mnist_source)
mnist_streamer.add_pipe(pipes.MnistDataParser())
mnist_streamer.add_pipe(pipes.ToTensor())
mnist_streamer.add_pipe(pipes.Batch(5))


def initiate_federated():
    for i in range(context.comm.size()):
        states = LogisticRegression(28 * 28, 10).state_dict()
        context.comm.send(i + 1, (i, states, 0, 0))


def init_server():
    server_comm = Streamer(ModelReceiver())
    # --> client_id, model_states, sample_size, round_id
    server_comm.add_pipe(pipes.ToModel(lambda: LogisticRegression(28 * 28, 10), model_pos=1))
    # --> client_id, model, sample_size, round_id
    server_comm.add_pipe(pipes.Collect(size=2))
    # --> [(client_id, lr_model, sample_size)*size], round_id
    server_comm.add_pipe(pipes.AvgAggregator())
    # --> global_model_state, training_size, round_id
    server_comm.add_pipe(pipes.ToModel(lambda: LogisticRegression(28 * 28, 10), model_pos=0))
    # --> global_model, training_size, round_id
    server_comm.add_pipe(pipes.Infer(
        model_pos=0, criterion=nn.CrossEntropyLoss(),
        test_data=mnist_streamer.next(id=100)[0]
    ))
    # --> global_model, training_size, round_id
    server_comm.add_pipe(pipes.ToModelDictState(model_pos=0))
    # --> global_model_state, training_size, round_id
    server_comm.add_pipe(pipes.ToRandomClients([1, 2, 3], nb_client=2))
    # --> global_model_state, training_size, round_id + 1
    return server_comm


def init_trainer():
    client_comm = Streamer(ModelReceiver())
    # --> client_id, model_states, sample_size, round_id
    client_comm.add_pipe(pipes.ToModel(lambda: LogisticRegression(28 * 28, 10), model_pos=1))
    # --> client_id, lr_models, sample_size, round_id
    client_comm.add_pipe(pipes.TrainModel(
        epochs=10, optimizer_method='sgd', data_streamer=mnist_streamer,
        criterion=nn.CrossEntropyLoss()
    ))
    # --> client_id, trained_lr_model_states,sample_size, round_id
    client_comm.add_pipe(pipes.ToModel(lambda: LogisticRegression(28 * 28, 10), model_pos=1))
    # --> client_id, trained_lr_model,sample_size, round_id
    client_comm.add_pipe(pipes.Infer(
        model_pos=1, criterion=nn.CrossEntropyLoss(),
        test_data=mnist_streamer.next(id=100)[0]
    ))
    # --> client_id, trained_lr_model,sample_size, round_id
    client_comm.add_pipe(pipes.ToModelDictState(model_pos=1))
    # --> client_id, trained_lr_model_states,sample_size, round_id
    client_comm.add_pipe(pipes.ToServer())
    # --> client_id, trained_lr_model_states,sample_size, round_id
    return client_comm


logger = logging.getLogger('main')
logger.error("samira")
server = init_server()
trainers = [init_trainer(), init_trainer()]
initiate_federated()
for trainer in trainers:
    trainer.next(context=context)
for _ in range(len(trainers)):
    server.next(context=context)
