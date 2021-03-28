import logging

import torch
from torch import nn
from fedml_api.fedavg.SQLProvider import SQLDataProvider
from fedml_api.fedavg.utils import transform_tensor_to_list


class FedAVGTrainer(object):
    def __init__(self, client_index, device, model,
                 args):
        self.provider = SQLDataProvider(args)
        self.client_index = client_index
        self.host = client_index
        self.batch_size = args.batch_size
        self.used_x = []
        self.round = 0
        self.train_local = self.provider.cache(client_index, False)
        self.local_sample_number = self.provider.size()
        self.device = device
        self.args = args
        self.model = model
        # logging.info(self.model)
        self.model.to(self.device)
        self.served_client = []
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        if self.args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)

    def update_model(self, weights):
        logging.info("update_model. client_index = %d" % self.client_index)
        self.model.load_state_dict(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.served_client.append(client_index)
        logging.info("update_dataset. client_index = %d" % self.client_index)
        self.train_local = self.provider.cache(client_index, False).batch(self.batch_size)
        self.local_sample_number = self.provider.size()

    def update_round(self, round):
        self.round = round

    def train(self):
        self.model.to(self.device)
        # change to train mode
        self.model.train()

        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(self.train_local):
                # logging.info(images.shape)
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                log_probs = self.model(x)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                stats = {'client': self.client_index, 'loss': sum(epoch_loss)}
                logging.info('(client {}. Local Training Epoch: {} '
                             '\tLoss: {:.6f}'.format(self.client_index, epoch, sum(epoch_loss) / len(epoch_loss)))

        weights = self.model.cpu().state_dict()
        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number
