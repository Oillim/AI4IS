from data_processing import load_data
import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3*32*32, 10)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 3072)
        out = self.linear(xb)
        out = F.softmax(out, dim=1)
        return out
    
def train(x_train):
    model = CIFAR10()
    outputs = model(x_train)
    return outputs

def evaluate(outputs, y_train):
    maxProbs, y_preds = torch.max(outputs, dim=1) #torch.max returns the max value itself (maxProbs) as well as the index of the prediction (preds)
    #calculate the accuracy of the model
    acc = torch.sum(y_train==y_preds).item() / len(y_train)
    print(maxProbs)
    print(y_preds)
    print(f'Accuracy of logistic regression model (no optimization): {acc}')

if __name__ == "__main__":
    train_data, train_labels = load_data("../Data/cifar-10-batches-py")
    print(f'trainning data shape: {train_data.shape}')
    outputs = train(train_data)
    evaluate(outputs, train_labels)