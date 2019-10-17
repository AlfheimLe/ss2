from torchvision.datasets import MNIST
from torch.utils.data import DataLoader,TensorDataset
from sklearn.datasets import load_svmlight_file
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#from datautils import get_mnist
def get_data(file_name):
    data = load_svmlight_file(file_name, multilabel=True)
    return data[0], data[1]

def generate(y, y_dim, n):
    new_y = np.zeros([n, y_dim])
    for i in range(n):
        for j in y[i]:
            new_y[i, int(j)] = 1
    return new_y

def get_dataloader(num_training=10000,num_labeled=400,batch_size=200, y_dim=10, file_name = '../DataSource/rcv1_topics_train.svm'):
    train_data, train_label = get_data(file_name)

    train_data = train_data.toarray().astype(np.float32)
    n, _ = train_data.shape
    min_max = MinMaxScaler()
    train_data = min_max.fit_transform(train_data)
    #train_data = torch.from_numpy(train_data)
    # generate tensor y [1 0 0 1]
    train_label = generate(train_label, y_dim, n).astype(np.float32)
    train_X, _, train_y, _ = train_test_split(train_data, train_label, test_size=(1-num_training/n), random_state=None)

    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(train_X, train_y, test_size=(1-num_labeled/num_training), random_state=None)

    X_labeled = torch.from_numpy(X_labeled)
    X_unlabeled = torch.from_numpy(X_unlabeled)
    y_labeled = torch.from_numpy(y_labeled)
    y_unlabeled = torch.from_numpy(y_unlabeled)

    num_labeled = X_labeled.size(0)

    dataset={}
    dataset['labeled_data'] = X_labeled
    dataset['unlabeled_data'] = X_unlabeled
    dataset['labeled_label'] = y_labeled
    dataset['unlabeled_label'] = y_unlabeled


    dataloader={}
    dataloader['labeled'] = DataLoader(TensorDataset(dataset['labeled_data'], dataset['labeled_label']),
                                       batch_size=num_labeled // (num_training // batch_size), shuffle=True,
                                       num_workers=4)

    dataloader['unlabeled'] = DataLoader(TensorDataset(dataset['unlabeled_data'], dataset['unlabeled_label']),
                                       batch_size=batch_size-num_labeled // (num_training // batch_size), shuffle=True,
                                       num_workers=4)

    dataloader['test'] = DataLoader(TensorDataset(dataset['unlabeled_data'], dataset['unlabeled_label']),
                                       batch_size=X_unlabeled.size(0), shuffle=True,
                                       num_workers=4)
    return dataloader


if __name__=='__main__':
    get_dataloader()