import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable

import math





# def get_average(list):
def get_average(num):
    sum = 0
    for i in range(len(num)):
        sum += num[i]
    return sum/len(num)
 
 

def get_range(num):
    return max(num) - min(num)
 
 

def mediannum(num):
    listnum = [num[i] for i in range(len(num))]
    listnum.sort()
    lnum = len(num)
    if lnum % 2 == 1:
        i = int((lnum + 1) / 2)-1
        return listnum[i]
    else:
        i = int(lnum / 2)-1
        return (listnum[i] + listnum[i + 1]) / 2
 

def get_variance(num):
    sum = 0
    average = get_average(num)
    for i in range(len(num)):
        sum += (num[i] - average)**2
    return sum/len(num)
 

def get_stddev(num):
    average = get_average(num)
    sdsq = sum( [(num[i] - average) ** 2 for i in range(len(num))] )
    stdev = (sdsq / (len(num) - 1)) ** .5
    return stdev
 

def get_n_moment(num,n):
    sum = 0
    for i in ange(len(num)):
        sum += num[i]**n
    return sum/len(num)




class MyLinear(nn.Module):
    def __init__(self, feature_dim, class_num):
        super(MyLinear, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.weights = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        nn.init.xavier_uniform_(self.weights)
    
    def forward(self, feat, label):
        # Calculate logits
        valuation_logits = torch.matmul(feat, torch.transpose(self.weights, 0, 1))
        train_logits = valuation_logits
        return valuation_logits, train_logits, self.weights





class MetricLogits(nn.Module):
    def __init__(self, feature_dim, class_num):
        super(MetricLogits, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.weights = nn.Parameter( torch.FloatTensor(class_num, feature_dim))
        nn.init.xavier_uniform_(self.weights)
        # nn.init.uniform_(self.weights, a=-1.0, b=1.0)
        # self.weights.requires_grad = False
        # self.weights = nn.Parameter(torch.rand(class_num, feature_dim))


    def forward(self, feat, label):
        diff = torch.unsqueeze(self.weights, dim=0) - torch.unsqueeze(feat, dim=1)
        diff = torch.mul(diff, diff)
        metric = torch.sum(diff, dim=-1)

        norm_features = torch.norm(feat, p=2, dim=-1, keepdim=True)
        norm_weights = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        diff_norm = torch.unsqueeze(norm_weights, dim=0) - torch.unsqueeze(norm_features, dim=1)
        diff_norm = torch.mul(diff_norm, diff_norm)
        norm_diff_sq_tables = torch.sum(diff_norm, dim=-1)


        y_onehot = torch.FloatTensor(metric.size(0), self.class_num)
        y_onehot.zero_()
        y_onehot = Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), 0.0001 * norm_diff_sq_tables)
        y_onehot = y_onehot + 1.0
        sq_p_dist = torch.mul(metric, y_onehot)
        # margin_logits = -0.5 * margin_dist


        distances = []
        for i in range(metric.size(0)):
            label_i = int(label[i])
            # print(cos[i, label_i])
            distance = metric[i, label_i].item()
            distances.append(distance)
        max_distance = max(distances)
        min_distance = min(distances) 
        avg_distance = get_average(distances)
        stdv_distance = get_stddev(distances)
        
        # print('Now stdv of distances is {:.4f}'.format(stdv_distance))
        # print('Now average distance is {:.2f}, max distance is {:.2f}, min distance is {:.2f}'.format(avg_distance, max_distance, min_distance))
        ############################## Theta ##############################
        # Calculate logits
        metric_mean = torch.mean(metric).item()
        metric_stdv = math.sqrt(torch.var(metric).item())
        # print('stdv of pos metric is {:.4f}'.format(stdv_distance))
        std_metric = metric - torch.mean(metric) #/ torch.var(metric) #metric_stdv

        max_stdmetric = torch.max(std_metric).item()
        min_stdmetric = torch.min(std_metric).item()

        valuation_logits = -1.0 * metric
        train_logits = -1.0 * std_metric
        # train_logits = -1.0 * sq_p_dist
        # train_logits = 1000.0 * (1.0 - 1.0 * std_metric)
        weights = self.weights
        return  train_logits, valuation_logits, weights















class PureKernalMetricLogits(nn.Module):
    def __init__(self, feature_dim, class_num):
        super(PureKernalMetricLogits, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.weights = nn.Parameter( torch.FloatTensor(class_num, feature_dim))
        self.scale = 2.0 * math.log(self.class_num - 1)
        nn.init.xavier_uniform_(self.weights)
        
    def forward(self, feat, label):
        # Calculating metric
        diff = torch.unsqueeze(self.weights, dim=0) - torch.unsqueeze(feat, dim=1)
        diff = torch.mul(diff, diff)
        metric = torch.sum(diff, dim=-1)
        kernal_metric = torch.exp(-1.0 * metric / 1.8)
        # Corresponding kernal metric calculating
        cor_metrics = []
        for i in range(kernal_metric.size(0)):
            label_i = int(label[i])
            distance = kernal_metric[i, label_i].item()
            cor_metrics.append(distance)
        avg_distance = get_average(cor_metrics)
        if avg_distance < 0.5:
            avg_distance = 0.5
        self.scale = (1.0/avg_distance) * math.log(self.class_num-1.0) #(get_average(Bs))
        # Return data
        train_logits = self.scale * kernal_metric
        return train_logits, kernal_metric














class KernalMetricLogits(nn.Module):
    def __init__(self, feature_dim, class_num):
        super(KernalMetricLogits, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.weights = nn.Parameter( torch.FloatTensor(class_num, feature_dim))
        self.scale = math.sqrt(2.0) * math.log(self.class_num - 1)
        nn.init.xavier_uniform_(self.weights)
        
    def forward(self, feat, label):
        # Calculating metric
        diff = torch.unsqueeze(self.weights, dim=0) - torch.unsqueeze(feat, dim=1)
        diff = torch.mul(diff, diff)
        metric = torch.sum(diff, dim=-1)
        kernal_metric = torch.exp(-0.01 * metric)

        # scale = 6.0

        # Corresponding kernal metric calculating
        cor_metrics = []
        for i in range(kernal_metric.size(0)):
            label_i = int(label[i])
            distance = kernal_metric[i, label_i].item()
            cor_metrics.append(distance)
        avg_distance = get_average(cor_metrics)

# ##########
        # probs = F.softmax(self.scale * kernal_metric).detach().cpu().numpy()
        # gt_probs = []
        # Bs = []
        # for i in range(kernal_metric.size(0)):
        #     # Prob
        #     gt_prob = probs[i, label_i]
        #     gt_probs.append(gt_prob)
        #     # B
        #     B = math.exp(self.scale * kernal_metric[i, label_i].data[0]) * (1.0/gt_prob - 1.0)
        #     Bs.append(B)
# ##########

        if avg_distance < 0.5:
            avg_distance = 0.5
        self.scale = (1.0/avg_distance) * math.log(self.class_num-1.0) #(get_average(Bs))
        self.scale = 1.0
        
        # Return data
        valuation_logits = self.scale * kernal_metric
        train_logits = self.scale * kernal_metric
        weights = self.weights
        return train_logits, valuation_logits, weights











class NormalizedInnerProductWithScale(nn.Module):
    """
    Paper:[COCOv2]
    Rethinking Feature Discrimination and Polymerization for Large scale recognition
    """
    def __init__(self, feature_dim, class_num, scale=20):
        super(NormalizedInnerProductWithScale, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.scale = scale
        self.weights = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, feat, label):
        # Unit vector for features
        norm_features = torch.norm(feat, p=2, dim=-1, keepdim=True)
        normalized_features = torch.div(feat, norm_features)
        # Unit vector for weights
        norm_weights = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        normalized_weights = torch.div(self.weights, norm_weights)

        # print('Avg weight norm is {:.6f}, avg feature norm i {:.6f}'.format((sum(norm_weights)/len(norm_weights)).item(), (sum(norm_features)/len(norm_features)).item() ) )
        # Normalized inner product, or cosine
        cos = torch.matmul(normalized_features, torch.transpose(normalized_weights, 0, 1))
        ############################## Norm ##############################
        avg_w_norm = (sum(norm_weights)/len(norm_weights)).item()
        avg_x_norm = (sum(norm_features)/len(norm_features)).item()
        print('Avg weight norm is {:.6f}, avg feature norm i {:.6f}'.format(avg_w_norm, avg_x_norm))
        ############################## Norm ##############################

        ############################## Theta ##############################
        thetas = []
        for i in range(cos.size(0)):
            label_i = int(label[i])
            theta = math.acos(cos[i, label_i].data[0]) / math.pi * 180  # degree
            thetas.append(theta)
        
        max_theta = max(thetas)
        min_theta = min(thetas) 
        avg_theta = get_average(thetas)
        stdv_theta = get_stddev(thetas)
        print('Now stdv of thetas is {:.4f}'.format(stdv_theta))

        # print('Now average theta is {:.2f}, max theta is {:.2f}, min theta is {:.2f}'.format(avg_theta, max_theta, min_theta))
        ############################## Theta ##############################
        # Calculate logits
        logits = self.scale * cos

        return cos, logits, avg_theta, min_theta, max_theta, stdv_theta, avg_w_norm, avg_x_norm




class CosFaceInnerProduct(nn.Module):
    """
    Paper:[CosFace] and [AM-Softmax]
    CosFace: Large Margin Cosine Loss for Deep Face Recognition;
    Additive Margin Softmax for Face Verification.
    """
    def __init__(self, feature_dim, class_num, scale=7.0, margin=0.15):
        super(CosFaceInnerProduct, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.scale = scale
        self.margin = margin
        self.weights = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, feat, label):
        # Unit vector for features
        norm_features = torch.norm(feat, p=2, dim=-1, keepdim=True)
        normalized_features = torch.div(feat, norm_features)
        # Unit vector for weights
        norm_weights = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        normalized_weights = torch.div(self.weights, norm_weights)
        # Normalized inner product, or cosine
        cos = torch.matmul(normalized_features, torch.transpose(normalized_weights, 0, 1))
        # Calculate logits
        logits = self.scale * cos
        # Calculate margin tables
        margin_tables = torch.zeros_like(cos)
        margin_tables = Variable(margin_tables).cuda()
        margin_tables.scatter_(1, torch.unsqueeze(label, dim=-1), self.margin)
        # Calculate marginal logits
        marginal_logits = self.scale * (cos - margin_tables)

        return marginal_logits, logits, self.weights
    



class ArcFaceInnerProduct(nn.Module):
    """
    Paper:[ArcFace]
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
    """
    def __init__(self, feature_dim, class_num, scale=30.0, margin=0.5, easy_margin=False):
        super(ArcFaceInnerProduct, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.scale = scale
        self.margin = margin
        self.weights = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        nn.init.xavier_uniform_(self.weights)
        # Setting behavior according to the value of easy_margin
        if easy_margin == -1: # If easy_margin = -1, just as the statement in the paper.
            self.Threshholder = -1
            self.out_indicator = 0
        elif easy_margin is True: # If easy_margin is True, for angle bigger than pi/2, do not add margin.
            self.Threshholder = 0
            self.out_indicator = 0
        else: # If easy_margin is Flase, do some adjustment for adding margin.
            self.Threshholder = - math.cos(self.margin)
            self.out_indicator = 1

    def forward(self, feat, label):
        # Unit vector for features
        norm_features = torch.norm(feat, p=2, dim=-1, keepdim=True)
        normalized_features = torch.div(feat, norm_features)
        # Unit vector for weights
        norm_weights = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        normalized_weights = torch.div(self.weights, norm_weights)
        # Normalized inner product, or cosine
        cos = torch.matmul(normalized_features, torch.transpose(normalized_weights, 0, 1))
        ############################## Norm ##############################
        avg_w_norm = (sum(norm_weights)/len(norm_weights)).item()
        avg_x_norm = (sum(norm_features)/len(norm_features)).item()
        # print('Avg weight norm is {:.6f}, avg feature norm i {:.6f}'.format(avg_w_norm, avg_x_norm))
        ############################## Norm ##############################
        innerproduct_logits = torch.matmul(feat, torch.transpose(self.weights, 0, 1))
        # Calculate logits
        logits = self.scale * cos
        # Calculate margin tables
        margin_tables = torch.zeros_like(cos)
        margin_tables = Variable(margin_tables).cuda()
        margin_tables_ext = torch.zeros_like(cos)
        margin_tables_ext = Variable(margin_tables_ext).cuda()
        thetas = []
        for i in range(cos.size(0)):
            label_i = int(label[i])
            theta = math.acos(cos[i, label_i].data[0]) / math.pi * 180  # degree
            thetas.append(theta)
    
        avg_theta = sum(thetas) / len(thetas)
        # avg_theta = mediannum(thetas)
        max_theta = max(thetas)
        min_theta = min(thetas) 
        stdv_theta = get_stddev(thetas)
        # print('Now average theta is {:.2f}, max theta is {:.2f}, min theta is {:.2f}'.format(avg_theta, max_theta, min_theta))

        for i in range(margin_tables.size(0)):
            label_i = int(label[i])
            if cos[i, label_i].item() > self.Threshholder:
                margin_tables[i, label_i] += self.margin
            else:
                margin_tables_ext[i, label_i] -= self.margin * math.sin(self.margin)
        # Calculate marginal logits
        margin_tables_ext *= self.out_indicator
        marginal_logits = self.scale * (torch.cos(torch.acos(cos) + margin_tables) + margin_tables_ext)
        # if return_ip:
        #     return cos, marginal_logits, innerproduct_logits
        # else:
        #     return cos, marginal_logits, avg_theta, min_theta, max_theta, stdv_theta, avg_w_norm, avg_x_norm
        return marginal_logits, self.scale * cos, self.weights







class PcheckArcFaceInnerProduct(nn.Module):
    """
    Paper:[ArcFace]
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition.
    """
    def __init__(self, feature_dim, class_num, scale=35.0, margin=0.5, easy_margin=False):
        super(PcheckArcFaceInnerProduct, self).__init__()
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.scale = scale
        self.margin = margin
        self.weights = nn.Parameter(torch.FloatTensor(class_num, feature_dim))
        nn.init.xavier_uniform_(self.weights)
        # Setting behavior according to the value of easy_margin
        if easy_margin == -1: # If easy_margin = -1, just as the statement in the paper.
            self.Threshholder = -1
            self.out_indicator = 0
        elif easy_margin is True: # If easy_margin is True, for angle bigger than pi/2, do not add margin.
            self.Threshholder = 0
            self.out_indicator = 0
        else: # If easy_margin is Flase, do some adjustment for adding margin.
            self.Threshholder = - math.cos(self.margin)
            self.out_indicator = 1

    def forward(self, feat, label, return_ip=False):
        # Unit vector for features
        norm_features = torch.norm(feat, p=2, dim=-1, keepdim=True)
        normalized_features = torch.div(feat, norm_features)
        # Unit vector for weights
        norm_weights = torch.norm(self.weights, p=2, dim=-1, keepdim=True)
        normalized_weights = torch.div(self.weights, norm_weights)
        # Normalized inner product, or cosine
        cos = torch.matmul(normalized_features, torch.transpose(normalized_weights, 0, 1))
        innerproduct_logits = torch.matmul(feat, torch.transpose(self.weights, 0, 1))
        # Calculate logits
        logits = self.scale * cos
        # Calculate margin tables
        margin_tables = torch.zeros_like(cos)
        margin_tables = Variable(margin_tables).cuda()
        margin_tables_ext = torch.zeros_like(cos)
        margin_tables_ext = Variable(margin_tables_ext).cuda()
        
        for i in range(margin_tables.size(0)):
            label_i = int(label[i])
            if cos[i, label_i].item() > self.Threshholder:
                margin_tables[i, label_i] += self.margin
            else:
                margin_tables_ext[i, label_i] -= self.margin * math.sin(self.margin)
        # Calculate marginal logits
        margin_tables_ext *= self.out_indicator
        marginal_logits = self.scale * (torch.cos(torch.acos(cos) + margin_tables) + margin_tables_ext)

        thetas = []
        probs = F.softmax(logits).detach().cpu().numpy()
        gt_probs = []
        Bs = []
        for i in range(cos.size(0)):
            # Theta
            label_i = int(label[i])
            theta = math.acos(cos[i, label_i].data[0]) / math.pi * 180  # degree
            thetas.append(theta)
            # Prob
            gt_prob = probs[i, label_i]
            gt_probs.append(gt_prob)
            # B
            B = math.exp(logits[i, label_i].data[0]) * (1.0/gt_prob - 1.0)
            Bs.append(B)
        # Thetas
        max_theta = max(thetas)
        min_theta = min(thetas) 
        avg_theta = get_average(thetas)
        med_theta = mediannum(thetas)
        # Ps
        max_p = max(gt_probs)
        min_p = min(gt_probs)
        avg_p = get_average(gt_probs)
        # B_avg_thetas
        B_avg = math.acos(math.log(get_average(Bs) / (self.class_num - 1)) / self.scale) / math.pi * 180 # Degree
        print('The largest P and it\'s theta is {:.6f} and {:.6f}'.format(max_p, min_theta))
        print('The smallest P and it\s theta is {:.6f} and {:.6f}'.format(min_p, max_theta))
        print('The average P is {:.6f}'.format(avg_p))
        print('B is {:.4f}'.format(B_avg))

        if return_ip:
            return cos, marginal_logits, innerproduct_logits
        else:
            # return cos, marginal_logits, avg_theta, med_theta, max_p, B_avg, 0, 0
            return cos, marginal_logits, avg_theta, med_theta, B_avg, self.scale, 0, 0




# class VarKernalMetricLogits(nn.Module):
#     def __init__(self, feature_dim, class_num):
#         super(VarKernalMetricLogits, self).__init__()
#         self.feature_dim = feature_dim
#         self.class_num = class_num
#         self.weights = nn.Parameter( torch.FloatTensor(class_num, feature_dim))
#         self.scale = 2.0 * math.log(self.class_num - 1)
#         nn.init.xavier_uniform_(self.weights)
        
#     def forward(self, feat, label):
#         # Calculating metric
#         diff = torch.unsqueeze(self.weights, dim=0) - torch.unsqueeze(feat, dim=1)
#         diff = torch.mul(diff, diff)
#         metric = torch.sum(diff, dim=-1)
#         # Corresponding Euchlidean metric 
#         cor_eu_metrics = []
#         for i in range(metric.size(0)):
#             label_i = int(label[i])
#             distance = torch.sqrt(metric[i, label_i]).item()
#             cor_eu_metrics.append(distance)

#         # 计算对应类别的原是距离方差
#         std_e_distance = get_variance(cor_eu_metrics)

#         # 计算所有类别的原是距离方差
#         std_all_e_distance = torch.var(torch.sqrt(metric)).item()
#         # 计算非对应类别的原是距离方差
#         std_nocor_e_distance = (self.class_num * std_all_e_distance - std_e_distance) / (self.class_num - 1)

#         std_tables = torch.ones_like(metric) * std_nocor_e_distance
#         std_tables = Variable(std_tables).cuda()
#         std_tables.scatter_(1, torch.unsqueeze(label, dim=-1), std_e_distance)

#         kernal_metric = torch.exp(-1.0 * metric / std_tables)
#         # Corresponding kernal metric calculating
#         cor_metrics = []
#         for i in range(kernal_metric.size(0)):
#             label_i = int(label[i])
#             distance = kernal_metric[i, label_i].item()
#             cor_metrics.append(distance)
#         avg_distance = get_average(cor_metrics)
#         # var_distance = get_variance(cor_metrics)
#         # print('The average corresponding metric is {:.4f}'.format(avg_distance))
#         # print('The variance cor metric is {:.4f}'.format(std_e_distance))
#         # print('The average corresponding eu metric is {:.4f}'.format(avg_e_distance))
#         # print('The variance non cor metric eu is {:.4f}'.format(std_nocor_e_distance))
#         if avg_distance < 0.5:
#             avg_distance = 0.5
#         self.scale = (1.0/avg_distance) * math.log(self.class_num-1.0) #(get_average(Bs))
#         # Return data
#         # train_logits = 3.0 * self.scale * kernal_metric
#         train_logits = 4.0 * kernal_metric

#         return train_logits, torch.exp(-1.0 * metric), self.weights
#         # return train_logits
