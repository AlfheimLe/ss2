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
    file_name = '../DataSource/yeast_train.svm'
    x, y = get_data(file_name)
    # numpy -> tensor
    n, x_dim = x.shape
    y_dim = 14
    z_dim = 128
    h_dim = []
    for i in range(y_dim):
        h_dim.append(128)
    worker_dim = 1

    x = x.toarray().astype(np.float32)
    min_max = MinMaxScaler()
    x = min_max.fit_transform(x)
    x = torch.from_numpy(x)
    #generate tensor y [1 0 0 1]
    y = torch.from_numpy(generate(y, y_dim, n).astype(np.float32))


    labeled_x = x[0:200, :]
    labeled_y = y[0:200, :]

    unlabeled_x = x[201:400, :]
    unlabeled_y = y[201:400, :]

    labeled_size = labeled_x.shape[0]
    unlabeled_size = unlabeled_x.shape[0]

    model = DSGM(args)
    print(model)
    #
    # trade-off parameter
    alpha = 0.1*unlabeled_size/labeled_size
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    m = unlabeled_size
    cuda = torch.cuda.is_available()
    if cuda:
        model = model.cuda()
    epoch_bar = tqdm(range(500))
    for epoch in epoch_bar:
        model.train()
        # print('training')
        total_loss, accuracy = (0., 0.)
        preds = torch.zeros_like(unlabeled_y)
        i = 0
        preds = torch.zeros_like(unlabeled_y).cuda()
        #batch_bar = tqdm(zip(range(m), cycle(labeled_x), cycle(labeled_y), unlabeled_x))
        #for i, batch_x, batch_y, batch_u in batch_bar:
        for i, batch_x, batch_y, batch_u in zip(range(m), cycle(labeled_x), cycle(labeled_y), unlabeled_x):
            batch_x, batch_y, batch_u = Variable(batch_x), Variable(batch_y), Variable(batch_u)
            worker_y = torch.zeros(args.workers, 1, batch_y.size(0))
            batch_x = batch_x.reshape(1, batch_x.size(0))
            batch_u = batch_u.reshape(1, batch_u.size(0))
            worker_y[0, :, :] = batch_y
            worker_y[1, :, :] = batch_y
            if cuda:
                batch_x, worker_y = batch_x.cuda(), worker_y.cuda()
                batch_u = batch_u.cuda()

            label_t, label_loss = model(batch_x, worker_y)
            unlabel_t, unlabel_loss = model(batch_u)
            workers_bar = worker_y.mean(dim=0)
            clas_loss = torch.sum(workers_bar * torch.log(label_t + 1e-6), dim=1).mean()
            J_alpha = label_loss+unlabel_loss-clas_loss

            J_alpha.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += J_alpha.item()
            preds[i, :] = unlabel_t
            # batch_bar.set_description('[Loss={:.4f}], [L_Loss={:.4f}], [U_Loss={:.4f}], [S_Loss={:.4f}]'.format(J_alpha.item(),
            #                                                                                                     -label_loss.item(),
            #                                                                                                     -unlabel_loss.item(),
            #                                                                                                     -clas_loss.item()))
            # preds[i, :] = logits
            # accuracy += torch.mean((torch.max(logits, 1)[1].data == torch.max(batch_y, 1)[1].data).float())
        accuracy = f1_score(unlabeled_y.cpu(), preds.cpu(), average='micro')

        epoch_bar.set_description("accuracy: {:.2f}, loss : {:.2f}\n".format(accuracy, total_loss/m))
        #print('loss: %f' % (total_loss/m) )

        # if epoch % 1 == 0:
        #     model.eval()
        #     m = unlabeled_size
        #     print("testing Epoch: {}".format(epoch))
        #     accuracy = 0.
        #     #print("[Train]\t\t J_a: {:.2f}, accuracy: {:.2f}".format(total_loss / m, accuracy / m))
        #
        #     total_loss, accuracy = (0, 0)
        #     preds = torch.zeros_like(unlabeled_y).cuda()
        #     for i, test_x, test_y in zip(range(m),unlabeled_x, unlabeled_y):
        #         test_x, test_y = Variable(test_x), Variable(test_y)
        #         test_x = test_x.reshape(1, test_x.size(0))
        #         test_y = test_y.reshape(1, test_y.size(0))
        #         if cuda:
        #             test_x, test_y = test_x.cuda(), test_y.cuda()
        #
        #         unlabel_t, unlabel_loss = model(test_x)
        #
        #         preds[i,:] = unlabel_t
        #
        #         # _, pred_idx = torch.max(logits, 1)
        #         # _, lab_idx = torch.max(test_y, 1)
        #     accuracy = f1_score( unlabeled_y.cpu(),preds.cpu(), average='micro')
        #
        #     print("accuracy: {:.2f}".format( accuracy ))

if __name__== '__main__':
    main()