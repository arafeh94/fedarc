import copy
import json
import logging
import os
import time
import torch
import numpy as np
from torch import nn
from fedml_api.fedavg.SQLProvider import SQLDataProvider
from fedml_api.fedavg.utils import compare_models
from fedml_api.model.linear.lr import LogisticRegression


# noinspection PyMethodMayBeStatic
class FedAVGAggregator(object):
    def __init__(self, worker_num, device, model, args):
        self.provider = SQLDataProvider(args)
        self.batch_size = args.batch_size
        self.worker_num = worker_num
        self.device = device
        self.args = args
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        self.model, _ = self.init_model(model)
        self.cached_model, _ = self.init_model(model)
        self.model_influence = {}

    def init_model(self, model):
        model_params = model.state_dict()
        # logging.info(model)
        return model, model_params

    def get_global_model_params(self):
        return self.model.state_dict()

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate_models(self, models_dict: dict):
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in models_dict.keys():
            model_list.append((self.sample_num_dict[idx], models_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def aggregate(self):
        averaged_params = self.aggregate_models(self.model_dict)
        self.model.load_state_dict(averaged_params)
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        num_clients = min(client_num_per_round, client_num_in_total)
        np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
        client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        print(client_indexes)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def test_on_all_clients(self, round_idx):
        self.test_model_on_all_clients(self.model, round_idx)

    def test_model_on_all_clients(self, model, round_idx):
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            logging.info("################local_test_on_all_clients : {}".format(round_idx))
            train_num_samples = []
            train_tot_corrects = []
            train_losses = []

            test_num_samples = []
            test_tot_corrects = []
            test_losses = []
            for client_idx in range(self.args.client_num_in_total):
                # train data
                train_data = self.provider.cache(client_idx).batch(self.batch_size)
                train_tot_correct, train_num_sample, train_loss = self._infer_model(model, train_data)
                train_tot_corrects.append(copy.deepcopy(train_tot_correct))
                train_num_samples.append(copy.deepcopy(train_num_sample))
                train_losses.append(copy.deepcopy(train_loss))

                # test data
                test_data = self.provider.cache(client_idx, True).batch(self.batch_size)
                test_tot_correct, test_num_sample, test_loss = self._infer_model(model, test_data)
                test_tot_corrects.append(copy.deepcopy(test_tot_correct))
                test_num_samples.append(copy.deepcopy(test_num_sample))
                test_losses.append(copy.deepcopy(test_loss))

                """Note: CI environment is CPU-based computing. The training speed for RNN training is to slow in 
                this setting, so we only test a client to make sure there is no programming error. """
                if self.args.ci == 1:
                    break

            # test on training dataset
            train_acc = sum(train_tot_corrects) / sum(train_num_samples)
            train_loss = sum(train_losses) / sum(train_num_samples)
            # wandb.log({"Train/Acc": train_acc, "round": round_idx})
            # wandb.log({"Train/Loss": train_loss, "round": round_idx})
            train_stats = {'training_acc': train_acc, 'training_loss': train_loss}
            logging.info(train_stats)

            # test on test dataset
            test_acc = sum(test_tot_corrects) / sum(test_num_samples)
            test_loss = sum(test_losses) / sum(test_num_samples)
            # wandb.log({"Test/Acc": test_acc, "round": round_idx})
            # wandb.log({"Test/Loss": test_loss, "round": round_idx})
            test_stats = {'test_acc': test_acc, 'test_loss': test_loss}
            logging.info(test_stats)
            return train_stats, test_stats

    def test_on_all_sub_models(self, round_idx: int):
        print("####round: " + str(round_idx) + "####")
        round_key = "round_" + str(round_idx)
        train_stats, test_stats = self.test_model_on_all_clients(self.model, round_idx)
        print("round", round_idx)
        print("global_model_train", train_stats)
        print("global_model_test", test_stats)
        self.model_influence[round_key] = {}
        self.model_influence[round_key]["."] = {"train_stats": train_stats, "test_stats": test_stats}
        print("model[.]:", train_stats)
        for idx in self.model_dict:
            temp_model_dict = dict(self.model_dict)
            del temp_model_dict[idx]
            model_params = self.aggregate_models(temp_model_dict)
            sub_model = LogisticRegression(28 * 28, 10)
            # sub_model = self.cached_model
            sub_model.load_state_dict(model_params)
            train_stats, test_stats = self.test_model_on_all_clients(sub_model, round_idx)
            print("model[" + str(idx) + "]:", train_stats)
            print("model[" + str(idx) + "].train", train_stats)
            print("model[" + str(idx) + "].test", test_stats)
            influence = self._influence(sub_model)
            print("model[" + str(idx) + "].influence", influence)
            influence_no_real, influence_both, influence_original = influence
            influence_ecl = self._influence_ecl(sub_model)
            self.model_influence[round_key][str(idx)] = {}
            # self.model_influence[round_key][str(idx)]["test_stats"] = train_stats
            self.model_influence[round_key][str(idx)]["train_stats"] = train_stats
            self.model_influence[round_key][str(idx)]["influence_no_real"] = influence_no_real
            self.model_influence[round_key][str(idx)]["influence_real"] = influence_both
            self.model_influence[round_key][str(idx)]["influence_ecl"] = influence_ecl.numpy()
            # print("influence[" + str(idx) + "]", influence)
            # print("euclidean_influence[" + str(idx) + "]", self._influence_ecl(sub_model))
            # plotter.append(influence)
        print("####end of round: " + str(round_idx) + "####")
        # plotter.save("round_" + str(round_idx))
        # self.log_cache.save()
        return ""

    def _influence_ecl(self, model):
        original = self.model.state_dict()
        sub = model.state_dict()
        l2_norm = torch.dist(original["linear.weight"], sub["linear.weight"], 2)
        return l2_norm

    # noinspection PyUnresolvedReferences
    def _influence(self, model):
        client_nums = self.args.influence_test_clients
        influence_no_labels = torch.tensor(0, dtype=torch.float)
        influence_correct_labels_both = torch.tensor(0, dtype=torch.float)
        influence_correct_labels_original = torch.tensor(0, dtype=torch.float)
        client_round = 0
        for client_idx in range(self.args.client_num_in_total):
            train_data = self.provider.cache(client_idx, True)
            if len(train_data) == 0:
                continue
            train_batches = train_data.batch(self.batch_size)

            deletion_prediction = self.predict(model, train_batches)
            original_prediction = self.predict(self.model, train_batches)
            deletion_labels = self._predictions_to_label(deletion_prediction)
            original_labels = self._predictions_to_label(original_prediction)
            real_labels = train_data.y

            # calculate the influence without taking into consideration the real labels
            influence_no_labels += self._influence_function_no_labels(
                deletion_prediction, original_prediction,
                deletion_labels, original_labels, real_labels
            )

            # calculate the influence taking into consideration only the correct labels in both predictions
            influence_correct_labels_both += self._influence_function_only_correct_labels_both(
                deletion_prediction, original_prediction,
                deletion_labels, original_labels, real_labels
            )

            # calculate the influence taking into consideration only the correct labels in original predictions
            influence_correct_labels_original += self._influence_function_only_correct_labels_original(
                deletion_prediction, original_prediction,
                deletion_labels, original_labels, real_labels
            )

            client_round += 1
            if 0 < client_nums <= client_round:
                break
        influence_no_labels = influence_no_labels.item() / client_nums
        influence_correct_labels_both = influence_correct_labels_both.item() / client_nums
        influence_correct_labels_original = influence_correct_labels_original.item() / client_nums
        return influence_no_labels, influence_correct_labels_both, influence_correct_labels_original

    def _influence_of_predictions(self, left, right):
        if len(left) == 0:
            return torch.tensor(0, dtype=torch.float)

        influence = torch.tensor(data=0.0, dtype=torch.float)
        for i in range(len(left)):
            difference = left[i] - right[i]
            difference = torch.abs(difference)
            influence += torch.mean(difference)
        return influence / len(left)

    def _conditional_influence_function(self, deletion_prediction, original_prediction,
                                        deletion_labels, original_labels, real_labels, condition):
        new_deletion_predictions = torch.tensor([])
        new_original_predictions = torch.tensor([])
        for index, label in enumerate(real_labels):
            if condition(deletion_labels[index], original_labels[index], real_labels[index]):
                new_deletion_predictions = torch.cat(
                    (new_deletion_predictions, torch.unsqueeze(deletion_prediction[index], 0)))
                new_original_predictions = torch.cat(
                    (new_original_predictions, torch.unsqueeze(original_prediction[index], 0)))
        return self._influence_of_predictions(new_deletion_predictions, new_original_predictions)

    def _influence_function_no_labels(self, deletion_prediction, original_prediction,
                                      deletion_labels, original_labels, real_labels):
        return self._conditional_influence_function(deletion_prediction, original_prediction,
                                                    deletion_labels, original_labels, real_labels,
                                                    lambda d, o, r: True)

    def _influence_function_only_correct_labels_both(self, deletion_prediction, original_prediction,
                                                     deletion_labels, original_labels, real_labels):

        return self._conditional_influence_function(deletion_prediction, original_prediction,
                                                    deletion_labels, original_labels, real_labels,
                                                    lambda d, o, r: d == r == o)

    def _influence_function_only_correct_labels_original(self, deletion_prediction, original_prediction,
                                                         deletion_labels, original_labels, real_labels):
        return self._conditional_influence_function(deletion_prediction, original_prediction,
                                                    deletion_labels, original_labels, real_labels,
                                                    lambda d, o, r: o == r)

    def _predictions_to_label(self, predictions):
        labels = torch.tensor([])
        for prediction in predictions:
            predicted_label = torch.argmax(prediction)
            labels = torch.cat((labels, predicted_label.reshape(1)))
        return labels

    def _infer(self, test_data):
        return self._infer_model(self.model, test_data)

    def predict(self, model, data):
        predictions = None
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(data):
                x = x.to(self.device)
                target.to(self.device)
                if predictions is None:
                    predictions = model(x)
                else:
                    predictions = torch.cat((predictions, model(x)))
        return predictions

    def _infer_model(self, model, test_data):
        model.eval()
        model.to(self.device)

        test_loss = test_acc = test_total = 0.
        criterion = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)
                pred = model(x)
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                test_acc += correct.item()
                test_loss += loss.item() * target.size(0)
                test_total += target.size(0)

        return test_acc, test_total, test_loss
