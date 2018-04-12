import numpy as np
import time
import cv2
import math
from scipy.signal import butter, lfilter, freqz, group_delay, filtfilt
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import cdist
from torch import np
import util
import torch
import torch as T
import torch.nn as nn
from torch.autograd import Variable
from config_reader import config_reader
# from time_probe import tic, toc, time_summary

param_, model_ = config_reader()

# USE_MODEL = model_['use_model']
USE_GPU = False #param_['use_gpu']
TORCH_CUDA = lambda x: x.cuda() if USE_GPU else x

head=0
weight_name = './model/pose_model.pth'


from os import listdir

print(listdir("./model"))

torch.set_num_threads(torch.get_num_threads())
blocks = {}

coordinates = ["x", "y"]
joints_list = ["right_shoulder", "right_elbow", "right_wrist", "left_shoulder","left_elbow", "left_wrist",
        "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
        "right_eye", "right_ear","left_eye", "left_ear", "nose ", "neck"]

index_shoulder=[0,3]
index_elbow=[1,4]
index_wrist=[2,5]
index_hip=[6,9]
index_knee=[7,10]
index_ankle=[8,11]
index_eye=[12,14]
index_ear=[13,15]

index_list=[index_shoulder,index_elbow,index_wrist,index_hip,index_knee,index_ankle,index_eye,index_ear]

important_joints = [0,3,6,7,8,9,10,11]
print(important_joints)
limb_list=['shoulder','elbow','wrist','hip','knee','ankle','Neck','eye','ear']

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
           [1,16], [16,18], [3,17], [6,18]]

# the middle joints heatmap correpondence
mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
          [55,56], [37,38], [45,46]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

block0  = [{'conv1_1':[3,64,3,1,1]},{'conv1_2':[64,64,3,1,1]},{'pool1_stage1':[2,2,0]},{'conv2_1':[64,128,3,1,1]},{'conv2_2':[128,128,3,1,1]},{'pool2_stage1':[2,2,0]},{'conv3_1':[128,256,3,1,1]},{'conv3_2':[256,256,3,1,1]},{'conv3_3':[256,256,3,1,1]},{'conv3_4':[256,256,3,1,1]},{'pool3_stage1':[2,2,0]},{'conv4_1':[256,512,3,1,1]},{'conv4_2':[512,512,3,1,1]},{'conv4_3_CPM':[512,256,3,1,1]},{'conv4_4_CPM':[256,128,3,1,1]}]

blocks['block1_1']  = [{'conv5_1_CPM_L1':[128,128,3,1,1]},{'conv5_2_CPM_L1':[128,128,3,1,1]},{'conv5_3_CPM_L1':[128,128,3,1,1]},{'conv5_4_CPM_L1':[128,512,1,1,0]},{'conv5_5_CPM_L1':[512,38,1,1,0]}]

blocks['block1_2']  = [{'conv5_1_CPM_L2':[128,128,3,1,1]},{'conv5_2_CPM_L2':[128,128,3,1,1]},{'conv5_3_CPM_L2':[128,128,3,1,1]},{'conv5_4_CPM_L2':[128,512,1,1,0]},{'conv5_5_CPM_L2':[512,19,1,1,0]}]

for i in range(2,7):
    blocks['block%d_1'%i]  = [{'Mconv1_stage%d_L1'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L1'%i:[128,128,7,1,3]},
{'Mconv5_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L1'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L1'%i:[128,38,1,1,0]}]
    blocks['block%d_2'%i]  = [{'Mconv1_stage%d_L2'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L2'%i:[128,128,7,1,3]},
{'Mconv5_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L2'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L2'%i:[128,19,1,1,0]}]

def make_layers(cfg_dict):
    layers = []
    for i in range(len(cfg_dict)-1):
        one_ = cfg_dict[i]
        for k,v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    one_ = list(cfg_dict[-1].keys())
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)

layers = []
for i in range(len(block0)):
    one_ = block0[i]
    for k,v in one_.items():
        if 'pool' in k:
            layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
            layers += [conv2d, nn.ReLU(inplace=True)]

models = {}
models['block0']=nn.Sequential(*layers)

for k,v in blocks.items():
    models[k] = make_layers(v)

class pose_model(nn.Module):
    def __init__(self,model_dict,transform_input=False):
        super(pose_model, self).__init__()
        self.model0   = model_dict['block0']
        self.model1_1 = model_dict['block1_1']
        self.model2_1 = model_dict['block2_1']
        self.model3_1 = model_dict['block3_1']
        self.model4_1 = model_dict['block4_1']
        self.model5_1 = model_dict['block5_1']
        self.model6_1 = model_dict['block6_1']

        self.model1_2 = model_dict['block1_2']
        self.model2_2 = model_dict['block2_2']
        self.model3_2 = model_dict['block3_2']
        self.model4_2 = model_dict['block4_2']
        self.model5_2 = model_dict['block5_2']
        self.model6_2 = model_dict['block6_2']

    def forward(self, x):
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2  = torch.cat([out1_1,out1_2,out1],1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3   = torch.cat([out2_1,out2_2,out1],1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4   = torch.cat([out3_1,out3_2,out1],1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5   = torch.cat([out4_1,out4_2,out1],1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6   = torch.cat([out5_1,out5_2,out1],1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1,out6_2


model = pose_model(models)
model.load_state_dict(torch.load(weight_name))
TORCH_CUDA(model)
model.float()
model.eval()

#param_, model_ = config_reader()

def handle_one(oriImg):
    tic = time.time()
 #   print 1

    # for visualize
#canvas = np.copy(oriImg)
    multiplier = [x * model_['boxsize'] / oriImg.shape[0] for x in param_['scale_search']]
    scale = model_['boxsize'] / float(oriImg.shape[0])

 #   print 2 ,time.time()-tic
    tic=time.time()
 #   print 3,time.time()-tic
    tic=time.time()
    e=1
    b=0
    len_mul=e-b
    multiplier=multiplier[b:e]
    heatmap_avg = TORCH_CUDA(torch.zeros((len(multiplier),19,oriImg.shape[0], oriImg.shape[1])))
    paf_avg = TORCH_CUDA(torch.zeros((len(multiplier),38,oriImg.shape[0], oriImg.shape[1])))
    # heatmap_avg = torch.zeros((len(multiplier),19,oriImg.shape[0], oriImg.shape[1])).cuda()
    # paf_avg = torch.zeros((len(multiplier),38,oriImg.shape[0], oriImg.shape[1])).cuda()
    toc =time.time()
 #   print 4,toc-tic

    tic = time.time()


    for m in range(len(multiplier)):

        tictic= time.time()
        scale = multiplier[m]
        # print("scale", scale)

        imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_['stride'], model_['padValue'])
        imageToTest_padded = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5
 #       print imageToTest_padded.shape
        feed = TORCH_CUDA(Variable(T.from_numpy(imageToTest_padded)))
        output1,output2 = model(feed)
 #       print time.time()-tictic,"first part"
        tictic=time.time()
        heatmap = TORCH_CUDA(nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])))(output2)

        # nearest neighbors
        paf = TORCH_CUDA(nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])))(output1)

        globals()['heatmap_avg_%s'%m] = heatmap[0].data
        globals()['paf_avg_%s'%m] = paf[0].data
        #heatmap_avg[m] = heatmap[0].data
        #paf_avg[m] = paf[0].data
  #      print 'loop', m ,' ',time.time()-tictic, "second part"

   # print 5 , time.time()-tic
    toc =time.time()
    #print 'time is %.5f'%(toc-tic)
    temp1=(heatmap_avg_0)#+heatmap_avg_1)/float(len_mul)
    temp2=(paf_avg_0)#+paf_avg_1)/float(len_mul)

    heatmap_avg = TORCH_CUDA(T.transpose(T.transpose(T.squeeze(temp1),0,1),1,2))
    paf_avg     = TORCH_CUDA(T.transpose(T.transpose(T.squeeze(temp2),0,1),1,2))
    # heatmap_avg = T.transpose(T.transpose(T.squeeze(temp1),0,1),1,2).cuda()
    # paf_avg     = T.transpose(T.transpose(T.squeeze(temp2),0,1),1,2).cuda()
    #heatmap_avg = T.transpose(T.transpose(T.squeeze(T.mean(heatmap_avg, 0)),0,1),1,2).cuda()
    #paf_avg     = T.transpose(T.transpose(T.squeeze(T.mean(paf_avg, 0)),0,1),1,2).cuda()
    heatmap_avg=heatmap_avg.cpu().numpy()
    paf_avg    = paf_avg.cpu().numpy()
    all_peaks = []
    peak_counter = 0
    #print '5bis', time.time()-toc
    toc =time.time()
    #maps =
#    s= heatmap_avg[:,:,0].shape
#    map_ori=heatmap_avg[:,:,12].cuda()
#    map = gaussian_filter(map_ori, sigma=3)
#
#    map_left = np.ones(map.shape)
#    map_left[1:,:] = map[:-1,:]
#    map_right = np.zeros(map.shape)
#    map_right[:-1,:] = map[1:,:]
#    map_up = np.zeros(map.shape)
#    map_up[:,1:] = map[:,:-1]
#    map_down = np.zeros(map.shape)
#    map_down[:,:-1] = map[:,1:]
#
#    peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param_['thre1']))
#
#    peaks0 = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
#
#    peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks0]
#    peaks_with_score_and_id0 = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
    for part in range(18):
        if part<18-head:
            map_ori = heatmap_avg[:,:,part]
            map = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map.shape)
            map_left[1:,:] = map[:-1,:]
            map_right = np.zeros(map.shape)
            map_right[:-1,:] = map[1:,:]
            map_up = np.zeros(map.shape)
            map_up[:,1:] = map[:,:-1]
            map_down = np.zeros(map.shape)
            map_down[:,:-1] = map[:,1:]

            peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param_['thre1']))

            peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
            globals()['peaks%s'%part]=peaks
            peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
            id = range(peak_counter, peak_counter + len(peaks_with_score))
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
            globals()['peaks_with_score_and_id_%s'%part]=  peaks_with_score_and_id
            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks_with_score)
        else :
            all_peaks.append(peaks_with_score_and_id_0)
            peak_counter += len(peaks0)


  #  print 6,time.time()-toc
    tic=time.time()
    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)-head):
        score_mid = paf_avg[:,:,[x-19-head for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]-1]
        candB = all_peaks[limbSeq[k][1]-1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    vec = np.divide(vec, norm)

                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                   np.linspace(candA[i][1], candB[j][1], num=mid_num))
                    startend = []
                    for elem in np.linspace(candA[i][0], candB[j][0], num=mid_num):
                        for elem1 in np.linspace(candA[i][1], candB[j][1], num=mid_num):
                            startend.append((elem, elem1))

                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                      for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                      for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    if norm==0:
                        score_with_dist_prior = sum(score_midpts)/len(score_midpts)
                    else:
                        score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
                    criterion1 = len(np.nonzero(score_midpts > param_['thre2'])[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])
 #   print 7, time.time()-tic
    tic=time.time()
    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    #print 'candidate is ', candidate
    for k in range(len(mapIdx)-head):

        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k]) - 1

            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    #print "found = 2"
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    """
    inserted in original Code from github because we want the function to return the raw data and not an image
    """
    output_coordinates=np.zeros((len(subset),18,2))
    for n in range(len(subset)):
        later = []
        for i in range(18):
            j = int(subset[n][i])
            if j==-1:
                continue
            output_coordinates[n,i-2,0]= candidate[j, 0]
            output_coordinates[n,i-2,1]= candidate[j, 1]
    # all_peaks = output_coordinates
    # print(all_peaks.shape)
    # for x in range(len(all_peaks)):
    #     for j in range(len(all_peaks[x])):
    #         cv2.circle(oriImg, (int(all_peaks[x,j,0]),int(all_peaks[x,j,1])) , 2, colors[x], thickness=-1)
    # a = np.random.randint(300)
    # cv2.imwrite(str(a)+".jpg", oriImg)
    return output_coordinates


# def handle_one(oriImg):
#     tic = time.time()
#  #   print 1
#
#     # for visualize
# #canvas = np.copy(oriImg)
#     multiplier = [x * model_['boxsize'] / oriImg.shape[0] for x in param_['scale_search']]
#
#     # scale = model_['boxsize'] / float(oriImg.shape[0])
#
#     #ind = np.argmin(np.absolute(np.array(multiplier)-1))
#     # scale = multiplier[ind]
#
#     #print("handle one1" ,time.time()-tic)
#     tic=time.time()
#  #   print 3,time.time()-tic
#     tic=time.time()
#     e=1
#     b=0
#     len_mul=e-b
#     multiplier=multiplier[b:e]
#     heatmap_avg = TORCH_CUDA(torch.zeros((len(multiplier),19,oriImg.shape[0], oriImg.shape[1])))
#     paf_avg = TORCH_CUDA(torch.zeros((len(multiplier),38,oriImg.shape[0], oriImg.shape[1])))
#     toc =time.time()
#     #print("handle one 2",toc-tic)
#
#     tic = time.time()
#
#
#     for m in range(len(multiplier)):
#         # print multiplier[m]
#         tictic= time.time()
#         scale = multiplier[m]
#         # print(scale)
#         imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
#         imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_['stride'], model_['padValue'])
#         imageToTest_padded = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5
#         # print imageToTest_padded.shape
#         feed = TORCH_CUDA(Variable(T.from_numpy(imageToTest_padded)))
#         output1,output2 = model(feed)
#  #       print time.time()-tictic,"first part"
#         tictic=time.time()
#         heatmap = TORCH_CUDA(nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])))(output2)
#         #nearest neighbors
#         paf = TORCH_CUDA(nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])))(output1)
#
#         globals()['heatmap_avg_%s'%m] = heatmap[0].data
#         globals()['paf_avg_%s'%m] = paf[0].data
#         #heatmap_avg[m] = heatmap[0].data
#         #paf_avg[m] = paf[0].data
#   #      print 'loop', m ,' ',time.time()-tictic, "second part"
#
#     #print("handle one 3" , time.time()-tic)
#     toc = time.time()
#     #print 'time is %.5f'%(toc-tic)
#     temp1=(heatmap_avg_0)#+heatmap_avg_1)/float(len_mul)
#     temp2=(paf_avg_0)#+paf_avg_1)/float(len_mul)
#     heatmap_avg = TORCH_CUDA(T.transpose(T.transpose(T.squeeze(temp1),0,1),1,2))
#     paf_avg     = TORCH_CUDA(T.transpose(T.transpose(T.squeeze(temp2),0,1),1,2))
#     #heatmap_avg = T.transpose(T.transpose(T.squeeze(T.mean(heatmap_avg, 0)),0,1),1,2).cuda()
#     #paf_avg     = T.transpose(T.transpose(T.squeeze(T.mean(paf_avg, 0)),0,1),1,2).cuda()
#     heatmap_avg=heatmap_avg.cpu().numpy()
#     paf_avg    = paf_avg.cpu().numpy()
#     all_peaks = []
#     peak_counter = 0
#     #print '5bis', time.time()-toc
#     #maps =
# #    s= heatmap_avg[:,:,0].shape
# #    map_ori=heatmap_avg[:,:,12].cuda()
# #    map = gaussian_filter(map_ori, sigma=3)
# #
# #    map_left = np.ones(map.shape)
# #    map_left[1:,:] = map[:-1,:]
# #    map_right = np.zeros(map.shape)
# #    map_right[:-1,:] = map[1:,:]
# #    map_up = np.zeros(map.shape)
# #    map_up[:,1:] = map[:,:-1]
# #    map_down = np.zeros(map.shape)
# #    map_down[:,:-1] = map[:,1:]
# #
# #    peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param_['thre1']))
# #
# #    peaks0 = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
# #
# #    peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks0]
# #    peaks_with_score_and_id0 = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
#     for part in range(18):
#         if part<18-head:
#             map_ori = heatmap_avg[:,:,part]
#             map = gaussian_filter(map_ori, sigma=3)
#
#             map_left = np.zeros(map.shape)
#             map_left[1:,:] = map[:-1,:]
#             map_right = np.zeros(map.shape)
#             map_right[:-1,:] = map[1:,:]
#             map_up = np.zeros(map.shape)
#             map_up[:,1:] = map[:,:-1]
#             map_down = np.zeros(map.shape)
#             map_down[:,:-1] = map[:,1:]
#
#             peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param_['thre1']))
#
#             peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
#             globals()['peaks%s'%part]=peaks
#             peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
#             id = range(peak_counter, peak_counter + len(list(peaks)))
#             peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
#             globals()['peaks_with_score_and_id_%s'%part]=  peaks_with_score_and_id
#             all_peaks.append(peaks_with_score_and_id)
#             peak_counter += len(list(peaks))
#         else :
#             all_peaks.append(peaks_with_score_and_id_0)
#             peak_counter += len(peaks0)
#
#
#     #print("handle one 4",time.time()-toc)
#     tic=time.time()
#     connection_all = []
#     special_k = []
#     mid_num = 10
#
#     for k in range(len(mapIdx)-head):
#         score_mid = paf_avg[:,:,[x-19-head for x in mapIdx[k]]]
#         candA = all_peaks[limbSeq[k][0]-1]
#         candB = all_peaks[limbSeq[k][1]-1]
#         nA = len(candA)
#         nB = len(candB)
#         indexA, indexB = limbSeq[k]
#         if(nA != 0 and nB != 0):
#             connection_candidate = []
#             for i in range(nA):
#                 for j in range(nB):
#                     vec = np.subtract(candB[j][:2], candA[i][:2])
#                     norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
#                     vec = np.divide(vec, norm)
#
#                     # startend = []
#                     # for elem1 in np.linspace(candA[i][0], candB[j][0], num=mid_num):
#                     #     for elem2 in np.linspace(candA[i][1], candB[j][1], num=mid_num):
#                     #         startend.append((elem1, elem2))
#                     startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
#                                    np.linspace(candA[i][1], candB[j][1], num=mid_num))
#
#                     vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
#                                       for I in range(len(startend))])
#                     vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
#                                       for I in range(len(startend))])
#
#                     score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
#                     if norm==0:
#                         score_with_dist_prior = sum(score_midpts)/len(score_midpts)
#                     else:
#                         score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)
#                     criterion1 = len(np.nonzero(score_midpts > param_['thre2'])[0]) > 0.8 * len(score_midpts)
#                     criterion2 = score_with_dist_prior > 0
#                     if criterion1 and criterion2:
#                         connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])
#
#             connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
#             connection = np.zeros((0,5))
#             for c in range(len(connection_candidate)):
#                 i,j,s = connection_candidate[c][0:3]
#                 if(i not in connection[:,3] and j not in connection[:,4]):
#                     connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
#                     if(len(connection) >= min(nA, nB)):
#                         break
#
#             connection_all.append(connection)
#         else:
#             special_k.append(k)
#             connection_all.append([])
#     #print("handle one 5", time.time()-tic)
#     tic=time.time()
#     # last number in each row is the total parts number of that person
#     # the second last number in each row is the score of the overall configuration
#     subset = -1 * np.ones((0, 20))
#     candidate = np.array([item for sublist in all_peaks for item in sublist])
#
#     #print 'candidate is ', candidate
#     for k in range(len(mapIdx)-head):
#
#         if k not in special_k:
#             partAs = connection_all[k][:,0]
#             partBs = connection_all[k][:,1]
#             indexA, indexB = np.array(limbSeq[k]) - 1
#
#             for i in range(len(connection_all[k])): #= 1:size(temp,1)
#                 found = 0
#                 subset_idx = [-1, -1]
#                 for j in range(len(subset)): #1:size(subset,1):
#                     if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
#                         subset_idx[found] = j
#                         found += 1
#
#                 if found == 1:
#                     j = subset_idx[0]
#                     if(subset[j][indexB] != partBs[i]):
#                         subset[j][indexB] = partBs[i]
#                         subset[j][-1] += 1
#                         subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
#                 elif found == 2: # if found 2 and disjoint, merge them
#                     j1, j2 = subset_idx
#                     #print "found = 2"
#                     membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
#                     if len(np.nonzero(membership == 2)[0]) == 0: #merge
#                         subset[j1][:-2] += (subset[j2][:-2] + 1)
#                         subset[j1][-2:] += subset[j2][-2:]
#                         subset[j1][-2] += connection_all[k][i][2]
#                         subset = np.delete(subset, j2, 0)
#                     else: # as like found == 1
#                         subset[j1][indexB] = partBs[i]
#                         subset[j1][-1] += 1
#                         subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
#
#                 # if find no partA in the subset, create a new subset
#                 elif not found and k < 17:
#                     row = -1 * np.ones(20)
#                     row[indexA] = partAs[i]
#                     row[indexB] = partBs[i]
#                     row[-1] = 2
#                     row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
#                     subset = np.vstack([subset, row])
