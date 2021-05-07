def batch(x, y, batch_size):
    if len(x) == 0:
        return list()
    batch_data = list()
    batch_size = len(x) if batch_size <= 0 or len(x) < batch_size else batch_size
    for i in range(0, len(x), batch_size):
        batched_x = x[i:i + batch_size]
        batched_y = y[i:i + batch_size]
        batch_data.append((batched_x, batched_y))
    return batch_data


def get_sample_size(batched_data):
    sample_size = 0
    for batch_idx, (x, labels) in enumerate(batched_data):
        x, labels = x, labels
        sample_size += len(x)
    return sample_size
