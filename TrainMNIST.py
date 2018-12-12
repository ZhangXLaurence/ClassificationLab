import torch
import torch.nn as nn
import torch.optim

import sys
import os
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

from bases.DataLoader import DataLoad
from bases.Models import SimpleNet
from bases.Losses import MarginInnerProduct
from Tools import ModelSaver

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize(feat, weights, labels, epoch):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i], markersize=0.1)
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    # plt.text(-4.8, 4.6, "epoch=%d" % epoch)
    plt.plot(weights[:,0], weights[:,1], '.', c='black', markersize=3)
    plt.savefig('./images/softmax_loss_epoch=%d.eps' % epoch,format='eps')
    plt.close()

def visualize3D(feat, labels, epoch):
    # data = np.random.randint(0, 255, size=[40, 40, 40])
    # x, y, z = data[0], data[1], data[2]
    ax = plt.subplot(111, projection='3d')
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    for i in range(10):
        ax.scatter(feat[labels == i, 0], feat[labels == i, 1], feat[labels == i, 2], c=c[i], s=0.1)
    # ax.scatter(x[:10], y[:10], z[:10], c='y', s=0.1)
    # ax.scatter(x[10:20], y[10:20], z[10:20], c='r', s=0.1)
    # ax.scatter(x[30:40], y[30:40], z[30:40], c='g', s=0.1)
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.savefig('./images/softmax_loss_3D_epoch=%d.eps' % epoch,format='eps')
    plt.close()





class TrainingModel(nn.Module):
    def __init__(self, inference_model, inner_product):
        super(TrainingModel, self).__init__()
        self.inference_model = inference_model
        self.inner_product = inner_product
    def forward(self, x, label):
        features = self.inference_model(x)
        evaluation_logits, train_logits, weights = self.inner_product(features, label)
        # logits = self.inner_product(features)
        return features, evaluation_logits, train_logits, weights
    def SaveInferenceModel():
        # TO BE DOWN
        return 0


def Test(test_loder, model):
    correct = 0
    total = 0
    for i, (data, target) in enumerate(test_loder):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        feats, valuation_logits, train_logits, weights = model(data, target)
        _, predicted = torch.max(valuation_logits.data, 1)
        total += target.size(0)
        correct += (predicted == target.data).sum()
    acc = (100. * float(correct)) / float(total)
    print('Test Accuracy on the {} test images:{}/{} ({:.2f}%) \n' .format(total, correct, total, acc))
    return acc



def Train(train_loader, model, criterion, optimizer, epoch, info_interval):
    ip1_loader = []
    idx_loader = []
    for i, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        feats, valuation_logits, train_logits, weights = model(data, target)
        loss = criterion[0](train_logits, target)

        _, predicted = torch.max(valuation_logits.data, 1)
        accuracy = (target.data == predicted).float().mean()

        optimizer[0].zero_grad()
        loss.backward()
        optimizer[0].step()
        ip1_loader.append(feats)
        idx_loader.append((target))

        if (i + 1) % info_interval == 0:
            print('Epoch [%d], Iter [%d/%d] Loss: %.4f Acc %.4f'
                  % (epoch, i + 1, len(train_loader) , loss.item(), accuracy))
        
    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    # visualize(feat.data.cpu().numpy(), weights.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)
    # visualize3D(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)
    

def Processing(NumEpoch, LrScheduler, Optimizer, train_loader, test_loder, model, criterion, info_interval, save_path):
    cur_best=0
    for epoch in range(NumEpoch):
        LrScheduler.step()
        print('Current Learning Rate: {}'.format(Optimizer.param_groups[0]['lr']))
        Train(train_loader, model, criterion, [Optimizer], epoch, info_interval)
        SavePath = save_path + str(epoch + 1) + '.model'
        ModelSaver.SaveModel(model, SavePath, epoch, 10)
        cur_acc = Test(test_loder, model)
        if cur_best < cur_acc:
            cur_best = cur_acc
        print('Current best test accuracy is {:.2f}% \n'.format(cur_best))



def main():
    ################################################################################################
    # This process, set up the whole models and parameters
    # Get Hyper parameters and sets

    # General arg
    arg_DeviceIds = [0]
    arg_NumEpoch = 50
    arg_InfoInterval = 100
    arg_SavePath = './checkpoints/softmax_MNIST_'
    arg_SaveEpochInterbal = 10

    # Data arg
    arg_TrainDataPath = './data'
    arg_TrainBatchSize = 128
    arg_TestBatchSize = 1024

    # arg_FeatureDim = 2
    arg_FeatureDim = 32
    arg_classNum = 10
    
    # Learning rate arg
    arg_BaseLr = 0.01
    arg_Momentum = 0.9
    arg_WeightDecay = 0.0000

    # Learning rate scheduler
    arg_LrEpochStep = 20
    arg_Gamma = 0.5

    # Dataset Loading
    TrainLoader, TestLoader = DataLoad.LoadMNIST(arg_TrainBatchSize, arg_TestBatchSize, arg_TrainDataPath)
    # TrainLoader, TestLoader = DataLoad.LoadFashionMNIST(arg_TrainBatchSize, arg_TestBatchSize, arg_TrainDataPath)

    # Model Constructing
    # Inference Model Constructing
    Inference = SimpleNet.SmallNet(feature_dim=arg_FeatureDim)
    # Innerproduct Construction
    # InnerProduct = torch.nn.Linear(arg_FeatureDim, arg_classNum)
    # InnerProduct = MarginInnerProduct.InnerProductWithScaleButNoUse(arg_FeatureDim, arg_classNum)
    InnerProduct = MarginInnerProduct.MetricLogits(arg_FeatureDim, arg_classNum)
    # InnerProduct = MarginInnerProduct.ArcFaceInnerProduct(arg_FeatureDim, arg_classNum, scale=7.0, margin=0.35)
    Model = torch.nn.DataParallel(TrainingModel(Inference, InnerProduct), arg_DeviceIds)

    # Losses and optimizers Defining
    # Softmax CrossEntropy
    SoftmaxLoss = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        SoftmaxLoss = SoftmaxLoss.cuda()
        Model = Model.cuda()
    criterion = [SoftmaxLoss]
    # Optimzer
    # Optimizer = torch.optim.SGD(Model.parameters(), lr=arg_BaseLr, momentum=arg_Momentum, weight_decay=arg_WeightDecay)
    Optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,Model.parameters()), lr=arg_BaseLr, momentum=arg_Momentum, weight_decay=arg_WeightDecay)
    
    # Learning rate Schedule
    LrScheduler = torch.optim.lr_scheduler.StepLR(Optimizer, arg_LrEpochStep, gamma=arg_Gamma)


    ################################################################################################

    # Resume from a checkpoint/pertrain

    # Training models
    # Testing models
    Processing(arg_NumEpoch, LrScheduler, Optimizer, TrainLoader, TestLoader, Model, criterion, arg_InfoInterval, arg_SavePath)



if __name__ == '__main__':
    main()
