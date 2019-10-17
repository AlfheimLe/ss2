import torch
import sys
import numpy as np
import torch.nn.functional as F
#from models import  Encoder, Decoder
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
#from inference import SVI, ImportanceWeightedSampler
from torch.autograd import Variable
from itertools import cycle
from sklearn.metrics import f1_score
#from models import DeepGenerativeModel
from get_args import process_args
from model import DSGM
sys.path.append('../../semi-supervised')
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from data import get_dataloader
#from models import DeepGenerativeModel

def generate(y, y_dim, n):
    new_y = np.zeros([n, y_dim])
    for i in range(n):
        for j in y[i]:
            new_y[i, int(j)] = 1
    return new_y

def binary_cross_entroy(r, x):
    return -torch.sum(x * torch.log(r+1e-8)+(1-x)*torch.log(1-r+1e-8), dim=-1)

def get_data(file_name):
    data = load_svmlight_file(file_name, multilabel=True)
    return data[0], data[1]

def main():
    args = process_args()

    dataloader = get_dataloader(num_training=1000,num_labeled=400,
                                batch_size=100, y_dim=args.y_dim,
                                file_name=args.input_file)

    #x, y = dataloader['labeled']

    model = DSGM(args)
    print(model)
    #
    # trade-off parameter
    #alpha = 0.1*unlabeled_size/labeled_size
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, betas=(0.9, 0.999))
    #m = unlabeled_size
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()
    epoch_bar = tqdm(range(100))

    # for epoch in epoch_bar:
    #     model.train()
    #     # print('training')
    #     total_loss, accuracy = (0., 0.)
    #     #preds = torch.zeros_like(unlabeled_y)
    #     i = 0
    #     #preds = torch.zeros_like(unlabeled_y).cuda()
    #     #batch_bar = tqdm(zip(range(m), cycle(labeled_x), cycle(labeled_y), unlabeled_x))
    #     #for i, batch_x, batch_y, batch_u in batch_bar:
    #     for batch_labeled in zip(dataloader['labeled']):
    #         batch_x, batch_y = batch_labeled[0]
    #         batch_x, batch_y = Variable(batch_x), Variable(batch_y)
    #
    #         ln, d = batch_y.size()
    #         worker_y = torch.zeros(args.workers, ln, d)
    #         #batch_x = batch_x.reshape(1, batch_x.size(0))
    #         #batch_u = batch_u.reshape(1, batch_u.size(0))
    #         worker_y[0, :, :] = batch_y
    #         worker_y[1, :, :] = batch_y
    #         if cuda:
    #             batch_x, worker_y = batch_x.cuda(), worker_y.cuda()
    #
    #         label_t, label_loss = model(batch_x, worker_y)
    #         workers_bar = worker_y.mean(dim=0)
    #         clas_loss = torch.mean(torch.sum(F.binary_cross_entropy( label_t, workers_bar, reduction='none'), dim=1))
    #         J_alpha = label_loss+clas_loss
    #
    #         J_alpha.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         #accuracy = f1_score(batch_uy.cpu(), (unlabel_t>0.5).float().cpu(), average='micro')
    #
    #         total_loss += J_alpha.item()
    #     if epoch % 1 == 0:
    #         model.eval()
    #         for test_batch in zip(dataloader['test']):
    #             test_x, test_y = test_batch[0]
    #             test_x = test_x.cuda()
    #             test_y = test_y.cuda()
    #             unlabel_t, unlabel_loss = model(test_x)
    #             unlabel_t = (unlabel_t>0.5).float()
    #             accuracy = f1_score(test_y.cpu(), unlabel_t.cpu(), average='macro')
    #     epoch_bar.set_description("f1: {:.2f}, loss : {:.2f}\n".format(accuracy, total_loss))


    for epoch in epoch_bar:
        model.train()
        # print('training')
        total_loss, accuracy = (0., 0.)
        #preds = torch.zeros_like(unlabeled_y)
        i = 0
        #preds = torch.zeros_like(unlabeled_y).cuda()
        #batch_bar = tqdm(zip(range(m), cycle(labeled_x), cycle(labeled_y), unlabeled_x))
        #for i, batch_x, batch_y, batch_u in batch_bar:
        for batch_labeled, batch_unlabeled in zip(dataloader['labeled'], dataloader['unlabeled']):
            batch_x, batch_y = batch_labeled
            batch_u, batch_uy = batch_unlabeled
            batch_x, batch_y, batch_u = Variable(batch_x), Variable(batch_y), Variable(batch_u)

            ln, d = batch_y.size()
            un, _ = batch_uy.size()
            worker_y = torch.zeros(args.workers, ln, d)
            #batch_x = batch_x.reshape(1, batch_x.size(0))
            #batch_u = batch_u.reshape(1, batch_u.size(0))
            worker_y[0, :, :] = batch_y
            worker_y[1, :, :] = batch_y
            if cuda:
                batch_x, worker_y = batch_x.cuda(), worker_y.cuda()
                batch_u = batch_u.cuda()

            label_t, label_loss = model(batch_x, worker_y)
            unlabel_t, unlabel_loss = model(batch_u)
            workers_bar = worker_y.mean(dim=0)
            clas_loss = torch.mean(torch.sum(F.binary_cross_entropy( label_t, workers_bar, reduction='none'), dim=1))
            J_alpha = label_loss+unlabel_loss+clas_loss

            J_alpha.backward()
            optimizer.step()
            optimizer.zero_grad()
            #accuracy = f1_score(batch_uy.cpu(), (unlabel_t>0.5).float().cpu(), average='micro')

            total_loss += J_alpha.item()
        if epoch % 1 == 0:
            model.eval()
            for test_batch in zip(dataloader['test']):
                test_x, test_y = test_batch[0]
                test_x = test_x.cuda()
                test_y = test_y.cuda()
                unlabel_t, unlabel_loss = model(test_x)
                unlabel_t = (unlabel_t>0.5).float()
                accuracy = f1_score(test_y.cpu(), unlabel_t.cpu(), average='macro')
        epoch_bar.set_description("f1: {:.2f}, loss : {:.2f}\n".format(accuracy, total_loss))

if __name__== '__main__':
    main()