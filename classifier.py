import math
import random

class Layer:
    def __init__(self, input_size, output_size, activation="relu"):
        self.weights = [[random.gauss(0, math.sqrt(2.0/input_size)) for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [0.0] * output_size
        self.activation = activation

    def forward(self, x):
        output = []
        for i in range(len(self.weights)):
            val = sum(w * xi for w, xi in zip(self.weights[i], x)) + self.biases[i]
            if self.activation == "relu":
                val = max(0, val)
            elif self.activation == "sigmoid":
                val = 1 / (1 + math.exp(-max(-500, min(500, val))))
            output.append(val)
        return output

class ImageClassifier:
    def __init__(self, input_size=784, num_classes=10, hidden_layers=None):
        if hidden_layers is None:
            hidden_layers = [128, 64]
        self.layers = []
        prev_size = input_size
        for h in hidden_layers:
            self.layers.append(Layer(prev_size, h, "relu"))
            prev_size = h
        self.layers.append(Layer(prev_size, num_classes, "sigmoid"))
        self.num_classes = num_classes

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        total = sum(x)
        if total > 0:
            x = [v / total for v in x]
        return x

    def evaluate(self, test_data, test_labels):
        correct = 0
        confusion = [[0]*self.num_classes for _ in range(self.num_classes)]
        for x, label in zip(test_data, test_labels):
            pred = self.predict(x)
            pred_class = pred.index(max(pred))
            if pred_class == label:
                correct += 1
            confusion[label][pred_class] += 1
        accuracy = correct / len(test_data) if test_data else 0
        return {"accuracy": accuracy, "confusion_matrix": confusion, "total": len(test_data)}

    def compute_metrics(self, confusion):
        metrics = {}
        for c in range(self.num_classes):
            tp = confusion[c][c]
            fp = sum(confusion[r][c] for r in range(self.num_classes)) - tp
            fn = sum(confusion[c]) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            metrics[c] = {"precision": precision, "recall": recall, "f1": f1}
        return metrics

if __name__ == "__main__":
    clf = ImageClassifier(input_size=16, num_classes=3, hidden_layers=[8, 4])
    sample = [random.random() for _ in range(16)]
    print("Prediction:", clf.predict(sample))
