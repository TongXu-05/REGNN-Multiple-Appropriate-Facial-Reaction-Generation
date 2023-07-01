import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import tqdm
from utils.meters import AverageMeter
from utils.loss import PCC, DistributionLoss, SoftDTW, PearsonCC, \
    AllMseLoss, ThreMseLoss, AllThreMseLoss, MidLoss, S_MSE
from utils.plot import plot_landmark
import random
import glob
import numpy as np
import os
from utils.evaluate import accelerated_dtw, ccc, person_r, pcc, s_mse
from utils.compute_distance_fun import compute_distance
from utils.file import mkdir_if_not_exist


class Trainer(object):
    def __init__(self, model, loss_name='DTW', cal_logdets=True, neighbors=None,
                 no_inverse=False, neighbor_pattern='nearest', num_frames=50,
                 stride=25, loss_mid=False):

        self.model = model
        if loss_name == 'Distribution':
            self.criterion = DistributionLoss()
        elif loss_name == 'DTW':
            self.criterion = SoftDTW(use_cuda=True)
        else:
            print('Use Neighbor Pattern: ' + neighbor_pattern)
            if neighbor_pattern == 'nearest':
                self.threshold = None
                self.criterion = AllThreMseLoss()
                self.mid_criterion = MidLoss()
            elif neighbor_pattern == 'all':
                self.threshold = None
                self.criterion = AllThreMseLoss(cal_type='min')
                self.mid_criterion = MidLoss(loss_type='L2')

        self.neighbors = neighbors
        self.cal_logdets = cal_logdets
        self.no_inverse = no_inverse
        self.neighbor_pattern = neighbor_pattern
        self.num_frames = num_frames
        self.stride = stride
        self.loss_mid = loss_mid

    def random_select(self, site_group_pid_clip_idx, test=False):
        if not test:
            dtype, site, group, pid, clip, idx = site_group_pid_clip_idx.split('+')
        else:
            site, group, pid, clip, idx = site_group_pid_clip_idx.split('+')

        site_group_pid_clip = '+'.join((site, group, pid, clip))
        neighbors = self.neighbors[site_group_pid_clip]
        if not test:
            neighbors = random.sample(neighbors, 10) if len(neighbors) > 10 else neighbors
            neighbors = [neighbor.split('+') + [idx] for neighbor in neighbors]
        else:
            neighbors = [neighbor.split('+') for neighbor in neighbors]
        return neighbors

    def load_npy(self, site_group_pid_clip_idx):
        if len(site_group_pid_clip_idx) == 5:
            site, group, pid, clip, idx = site_group_pid_clip_idx
        else:
            site, group, pid, clip = site_group_pid_clip_idx
            idx = -1

        idx = int(idx)
        pth = os.path.join('/root/autodl-tmp/Action-Generation/data/FinalData',
                           '*', 'npy', site, group, pid, clip + '.npy')

        npy_pth = glob.glob(pth)[0]
        npy_inputs = np.load(npy_pth)[idx: idx + self.num_frames] if not idx == -1 else np.load(npy_pth)
        return torch.from_numpy(npy_inputs).float().cuda()

    def _parse_data(self, data, test=False):
        v_inputs, a_inputs, targets = data
        v_inputs = v_inputs.cuda()
        a_inputs = a_inputs.cuda()
        if not test and self.neighbor_pattern == 'pair':
            return v_inputs, a_inputs, targets.cuda()

        all_neighbors = []
        lengthes = []
        if not test:
            if self.neighbor_pattern == 'nearest':
                for site_group_pid_clip_idx in targets:
                    neighbors = self.random_select(site_group_pid_clip_idx)
                    neighbors_npy = [None] * 10
                    for i, neighbor in enumerate(neighbors):
                        neighbors_npy[i] = self.load_npy(neighbor)
                    all_neighbors.append(neighbors_npy)
            elif self.neighbor_pattern == 'all':
                for site_group_pid_clip_idx in targets:
                    neighbors = self.random_select(site_group_pid_clip_idx)
                    lengthes.append(len(neighbors))
                    for neighbor in neighbors:
                        neighbor_val = self.load_npy(neighbor)
                        all_neighbors.append(neighbor_val)

                all_neighbors = torch.stack(all_neighbors, dim=0).transpose(2, 1)
        else:
            base_site_group_pid_clip_idx = targets[0].split('+')
            for i, _site_group_pid_clip_idx in enumerate(targets):
                _site_group_pid_clip_idx = _site_group_pid_clip_idx.split('+')
                if not _site_group_pid_clip_idx[:-1] == base_site_group_pid_clip_idx[:-1] or not \
                        int(_site_group_pid_clip_idx[-1]) == self.stride*i:
                    print('Not a single clip')
                    print(_site_group_pid_clip_idx)
                    print(aaa)
            site_group_pid_clip_idx = targets[0]
            all_neighbors = self.random_select(site_group_pid_clip_idx, test)
        return v_inputs, a_inputs, all_neighbors, lengthes

    def train(self, epoch, dataloader, optimizer, print_freq=1, train_iters=100):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_dtw = AverageMeter()
        losses_mid = AverageMeter()
        losses_det = AverageMeter()
        since = time.time()

        for i, data in enumerate(dataloader):

            v_inputs, a_inputs, targets, lengthes = self._parse_data(data)
            data_time.update(time.time() - since)
            torch.autograd.set_detect_anomaly(True)
            if not self.no_inverse:
                speaker_features, listener_features, params, edge, nearest_targets, loss_det = \
                    self.model(v_inputs, a_inputs, targets, lengthes)
            else:
                speaker_features, listener_features, loss_det = self.model(v_inputs, a_inputs, targets, lengthes)

            if self.neighbor_pattern == 'all':
                loss_dtw = self.criterion(speaker_features, listener_features, lengthes, threshold=self.threshold)
                if self.loss_mid:
                    loss_mid = self.mid_criterion(listener_features, lengthes)
                else:
                    loss_mid = 0.0
            else:
                loss_dtw = self.criterion(speaker_features, listener_features, lengthes, threshold=self.threshold)

            loss = loss_dtw + (loss_det if self.cal_logdets else 0.) + loss_mid

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_dtw.update(loss_dtw.item())
            losses_mid.update(loss_mid.item() if not type(loss_mid) == float else 0.)
            losses_det.update(loss_det.item() if not type(loss_det) == float else 0.)
            losses_pcc.update(loss_pcc.item() if not type(loss_pcc) == float else 0.)
            batch_time.update(time.time() - since)
            since = time.time()

            torch.set_printoptions(precision=4, sci_mode=False)
            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_DTW {:.3f} ({:.3f})\t'
                      'Loss_MID {:.3f} ({:.3f})\t'
                      'Loss_DET {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_dtw.val, losses_dtw.avg,
                              losses_mid.val, losses_mid.avg,
                              losses_det.val, losses_det.avg,
                              ))
                print('-' * 160)

            if i == train_iters:
                self.threshold = torch.tensor([(losses_dtw.avg / 3)]).cuda()
                print('='*10 + ' threshold ' + '='*10)
                print(self.threshold)
                break


    def test(self, testloader, print_freq=1, modify=False, save_base=None):
        self.model.eval()
        test_iters = len(testloader)
        self.modify = modify
        for data in tqdm(testloader):
            v_inputs, a_inputs, targets, _ = self._parse_data(data, test=True)
            base_site_group_pid_idx = data[2][0].split('+')
            save_path = os.path.join(save_base, os.path.join(*base_site_group_pid_idx[:-1]))
            mkdir_if_not_exist(save_path)

            with torch.no_grad():
                for j in range(SAMPLE_NUMS):
                    predictions = self.model.inverse(v_inputs, a_inputs, cal_norm=cal_norm, threshold=self.threshold)
                    if self.modify:
                        predictions = self.modify_outputs(predictions)
                    # 750 25
                    predictions = self.combine_preds(predictions)
                    torch.save(predictions, os.path.join(save_path, 'result-' + str(j) + '.pth'))


    def combine_preds(self, predictions):
        # 750 // stride, num_frames, 25
        if self.stride == self.num_frames:
            return predictions.reshape(predictions.shape[0]*predictions.shape[1], -1)

        nums = predictions.shape[0]
        all_predictions = [predictions[0][:self.stride]]
        last_pred = predictions[0][self.stride:]

        for i in range(1, nums):
            pred = predictions[i]
            all_predictions.append((last_pred + pred[:self.stride]) / 2)
            last_pred = pred[self.stride:]

        all_predictions.append(last_pred)
        all_predictions = torch.cat(all_predictions, dim=0)

        return all_predictions


    def modify_outputs(self, outputs):
        modified_outputs = outputs.clone().detach()
        modified_outputs[:, :15, :] = torch.where(modified_outputs[:, :15, :] >= 1.0, 1.0, 0.0)

        return modified_outputs

