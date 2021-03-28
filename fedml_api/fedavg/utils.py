import json

import torch
import numpy as np

def compare_models(model_1, model_2):
    """
    @param model_1:
    @param model_2:
    @return: 0 if 2 models are equals, the difference count otherwise
    """
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
    return models_differ


def transform_list_to_tensor(model_params_list):
    for k in model_params_list.keys():
        model_params_list[k] = torch.from_numpy(np.asarray(model_params_list[k])).float()
    return model_params_list


def transform_tensor_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().numpy().tolist()
    return model_params

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def json_save(obj, name="logs"):
    print("save logs")
    with open("./logs/" + name + ".json", "w") as f:
        json.dump(obj, f, cls=NumpyEncoder)
