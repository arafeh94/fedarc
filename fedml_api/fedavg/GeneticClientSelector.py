import numpy as np
from typing import List


class FedAvgClientSelector:
    class UpdateModel:
        def __init__(self, client_id, model, influence, round_idx):
            self.round = round_idx
            self.influence = influence
            self.model = model
            self.client_id = client_id

    class WheelSelector:
        def __init__(self, client_models):
            pass

    def __init__(self, args, clients_idx: List[int]):
        """
        Args:
            args: application arguments (how many client will be selected per round is important
            clients_idx: all available client_idx
        Params:
            client_models: a dictionary of {client id->[client model,influence,round]}
        """
        self.args = args
        self.clients_idx = clients_idx
        self.client_models = {}
        self.selected_clients = []
        self.round_idx = 0

    def update(self, updated_models: [UpdateModel]):
        for updated_model in updated_models:
            if updated_model.client_id not in self.client_models:
                self.client_models[updated_model.client_id] = {
                    "influence": updated_model.influence,
                    "rounds": 1
                }
            else:
                hold = self.client_models[updated_model.client_id]
                new_influence = (hold["influence"] + updated_model.influence) / 2
                new_rounds = hold["influence"] + 1
                self.client_models[updated_model.client_id] = {
                    "influence": new_influence,
                    "rounds": new_rounds
                }

    def generate(self, round_idx) -> [int]:
        self.round_idx = round_idx
        np.random.seed(self.round_idx)
        if not self.client_models:
            self._initial_selection()
            return self.selected_clients
        self._selection()
        self._crossover()
        self._mutation()
        return self.selected_clients

    def _initial_selection(self):
        positions = np.random.choice(range(len(self.clients_idx)), self.args.num_clients, replace=False)
        for pos in positions:
            self.selected_clients.append(self.clients_idx[pos])

    def _selection(self):
        pass

    def _crossover(self):
        pass

    def _mutation(self):
        pass
