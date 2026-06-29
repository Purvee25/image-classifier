import random

def train_test_split(data, labels, test_ratio=0.2):
    combined = list(zip(data, labels))
    random.shuffle(combined)
    split = int(len(combined) * (1 - test_ratio))
    train = combined[:split]
    test = combined[split:]
    return [x[0] for x in train], [x[1] for x in train], [x[0] for x in test], [x[1] for x in test]

def normalize(data):
    result = []
    for sample in data:
        min_val = min(sample)
        max_val = max(sample)
        rng = max_val - min_val if max_val != min_val else 1
        result.append([(x - min_val) / rng for x in sample])
    return result

def one_hot_encode(labels, num_classes):
    return [[1 if i == l else 0 for i in range(num_classes)] for l in labels]
