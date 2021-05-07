import copy
import json
import logging
import random

import numpy
import numpy as np
import torch
from torch import nn

from src.core import tools, cflog
from src.core.aoi.abs_streamer import Pipe, Streamer
from src.core.context import Context

logger = logging.getLogger('pipes')


class MnistDataParser(Pipe):

    def next(self, data, raw, **kwargs) -> object:
        x = []
        y = []
        for row in data:
            x.append(json.loads(row[2]))
            y.append(row[3])
        return x, y


class ToTensor(Pipe):
    def next(self, data, raw, **kwargs) -> object:
        x, y = data
        return torch.from_numpy(np.asarray(x)).float(), torch.from_numpy(np.asarray(y)).long()


class Batch(Pipe):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def next(self, data, raw, **kwargs) -> object:
        x, y = data
        return tools.batch(x, y, self.batch_size)


class ToModel(Pipe):
    def __init__(self, create_model: callable, model_pos):
        super().__init__()
        self.model: torch.nn.Module = create_model()
        self.model_pos = model_pos

    def next(self, data, raw, **kwargs) -> object:
        cp = [i for i in data]
        self.model.load_state_dict(data[self.model_pos])
        cp[self.model_pos] = self.model
        return tuple(cp)


class ToModelDictState(Pipe):
    def __init__(self, model_pos):
        super().__init__()
        self.model_pos = model_pos

    def next(self, data, raw, **kwargs) -> object:
        cp = [i for i in data]
        cp[self.model_pos] = data[self.model_pos].cpu().state_dict()
        return tuple(cp)


class Collect(Pipe):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.models = []

    def next(self, data, raw, **kwargs) -> object:
        client_id, model, sample_size, round_id = data
        self.models.append((client_id, model, sample_size))
        if len(self.models) == self.size:
            return self.models, round_id
        return None


class AvgAggregator(Pipe):
    def __init__(self):
        super().__init__()

    def next(self, data, raw, **kwargs) -> object:
        models, round_id = data
        clients_model = numpy.array(models)
        model_list = []
        training_size = 0

        for item in clients_model:
            client_id, model, sample_size = item
            model_list.append((sample_size, copy.deepcopy(model.state_dict())))
            training_size += sample_size
        (num0, averaged_params) = model_list[0]

        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_size
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params, training_size, round_id


class TrainModel(Pipe):
    def __init__(self, data_streamer: Streamer, optimizer_method: str, criterion, epochs: int, learn_rate=0.1,
                 weight_decay=0.1):
        super().__init__()
        self.epochs = epochs
        self.data_streamer = data_streamer
        self.optimizer_method = optimizer_method
        self.criterion = criterion
        self.learn_rate = learn_rate
        self.weight_decay = weight_decay

    def next(self, data, raw, **kwargs) -> object:
        client_id, model, sample_size, round_id = data
        logger.debug('client_id:', client_id, 'sample_size:', sample_size, 'round_id:', round_id)
        optimizer = self.build_optimizer(model)
        model.train()
        train_local, _ = self.data_streamer.next(id=client_id)
        sample_size = tools.get_sample_size(train_local)
        epoch_loss = []
        for epoch in range(self.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_local):
                x, labels = x, labels
                optimizer.zero_grad()
                log_probabilities = model(x)
                loss = self.criterion(log_probabilities, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                loss_total = sum(epoch_loss)
                logger.debug("total loss:", loss_total)
        weights = model.cpu().state_dict()
        return client_id, weights, sample_size, round_id

    def build_optimizer(self, model):
        if self.optimizer_method == "sgd":
            return torch.optim.SGD(model.parameters(), lr=self.learn_rate)
        else:
            return torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.learn_rate, weight_decay=self.weight_decay, amsgrad=True)


class ToRandomClients(Pipe):
    def __init__(self, all_clients, nb_client):
        super().__init__()
        self.nb_client = nb_client
        self.all_clients = all_clients

    def next(self, data, raw, **kwargs) -> object:
        context: Context = kwargs.get('context')
        all_clients = self.all_clients
        global_model_state, training_size, round_id = data
        random_selected_clients = np.random.choice(all_clients, self.nb_client)
        logger.debug("clients selected for next round:", random_selected_clients)
        for index, selected_client in enumerate(random_selected_clients):
            # index + 1 because index 0 is reserved for the server
            context.comm.send(index + 1, (selected_client, global_model_state, 0, round_id + 1))
        return data


class Infer(Pipe):
    def __init__(self, model_pos, criterion, test_data=None, stream_source: Streamer = None):
        super().__init__()
        self.test_data = test_data
        self.stream_source = stream_source
        self.model_pos = model_pos
        self.criterion = criterion
        if self.test_data is None and self.stream_source is None:
            raise Exception("no data passed for testing. fill either static_test_data, or stream_source")

    def next(self, data, raw, **kwargs) -> object:
        model = data[self.model_pos]
        if self.test_data is None:
            self.test_data = self.stream_source.next(**kwargs)
        model.eval()

        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(self.test_data):
                x = x
                target = target
                pred = model(x)
                loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)
        logger.debug('-test_acc:', test_acc, '-test_total:', test_total, '-test_loss:', test_loss)
        return data


class ToServer(Pipe):
    def __init__(self, server_pid=0):
        super().__init__()
        self.server_pid = server_pid

    def next(self, data, raw, **kwargs) -> object:
        client_id, model_states, sample_size, round_id = data
        logger.debug("sending data to server from client_id:", client_id)
        context: Context = kwargs.get('context')
        context.comm.send(self.server_pid, (client_id, model_states, sample_size, round_id))
        return data
