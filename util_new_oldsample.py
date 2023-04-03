from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import random
import pdb
def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        # for sample in self.dataset:
        for batch_idx, sample in enumerate(self.dataset):
            audio, video, video_st, total_label, audio_label, visual_label = sample['audio'].to('cuda'), sample['video_s'].to('cuda'), sample['video_st'].to('cuda'), sample['total_label'].type(torch.FloatTensor).to('cuda'), sample['audio_label'].type(torch.FloatTensor).to('cuda'), sample['visual_label'].type(torch.FloatTensor).to('cuda')  #[16,10,128] [16,80,2048], [16,10,512]
            self.model.zero_grad()
            # audio, video, video_st = torch.from_numpy(sample['audio']).to('cuda'), torch.from_numpy(sample['video_s']).to('cuda'), torch.from_numpy(sample['video_st']).to('cuda')#[16,10,128] [16,80,2048], [16,10,512]
            # audio = audio.unsqueeze(0)
            # video = video.unsqueeze(0)
            # video_st = video_st.unsqueeze(0)
            output, a_prob, v_prob, _ = self.model(audio, video, video_st)
            # input = variable(input)
            # output = self.model(input).view(1, -1)
            # label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), total_label.long())
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad == None:
                    precision_matrices[n].data += 0
                else:

                    precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss, loss1, loss2, loss3 = 0, 0, 0, 0
        # all
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        # ewc on different module
        part1 = {"fc_a","fc_v","fc_st","fc_fusion"}
        part3 = {"fc_prob"}
        # for n, p in model.named_parameters():
        #     if n.split(".")[0] in part1:
        #         _loss1 = self._precision_matrices[n] * (p - self._means[n]) ** 2
        #         loss1 = _loss1.sum()
        # for n, p in model.named_parameters():
        #     if (n.split(".")[0] not in part1) and (n.split(".")[0] not in part3):
        #         _loss2 = self._precision_matrices[n] * (p - self._means[n]) ** 2
        #         loss2 = _loss2.sum()
        #  for n, p in model.named_parameters():
        #     if n.split(".")[0] in part3:
        #         _loss3 = self._precision_matrices[n] * (p - self._means[n]) ** 2
        #         loss3 = _loss3.sum()
        # loss = loss1 + loss2 + loss3
        return loss


def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader):
    model.train()
    epoch_loss = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)


# def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
#               ewc: EWC, importance: float):
#     model.train()
#     epoch_loss = 0
#     for input, target in data_loader:
#         input, target = variable(input), variable(target)
#         optimizer.zero_grad()
#         output = model(input)
#         loss = F.cross_entropy(output, target) + importance * ewc.penalty(model)
#         epoch_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     return epoch_loss / len(data_loader)
def ewc_train(args, model, train_loader, optimizer, criterion, epoch, ewc: EWC, importance: float, old_data):
    model.train()


    for batch_idx, sample in enumerate(train_loader):

        audio, video, video_st, total_label, audio_label, visual_label = sample['audio'].to('cuda'), sample['video_s'].to('cuda'), sample['video_st'].to('cuda'), sample['total_label'].type(torch.FloatTensor).to('cuda'), sample['audio_label'].type(torch.FloatTensor).to('cuda'), sample['visual_label'].type(torch.FloatTensor).to('cuda')  #[16,10,128] [16,80,2048], [16,10,512]
        optimizer.zero_grad()
        output, a_prob, v_prob, _ = model(audio, video, video_st)
        # output.clamp_(min=1e-7, max=1 - 1e-7)
        # a_prob.clamp_(min=1e-7, max=1 - 1e-7)
        # v_prob.clamp_(min=1e-7, max=1 - 1e-7)

        # label smoothing
        a = 1.0
        v = 0.9 
        # Pa = a * target + (1 - a) * 0.5
        # Pv = v * target + (1 - v) * 0.5

        # individual guided learning
        # loss = criterion(output, total_label.long()) 
        class_loss = criterion(output, total_label.long())
        ewc_loss = importance * ewc.penalty(model)

        loss = class_loss + ewc_loss
        # print("class loss is {}, ewc loss is {}".format(class_loss, ewc_loss))
        current_sample =  random.sample(old_data, k=len(total_label))
        Flag = True
        for i in range(len(total_label)):
            if Flag:
                current_each_sample = current_sample[i]
                audio_old, video_old, video_st_old, total_label_old = torch.from_numpy(current_each_sample['audio']).to('cuda'), torch.from_numpy(current_each_sample['video_s']).to('cuda'), torch.from_numpy(current_each_sample['video_st']).to('cuda'), torch.tensor(current_each_sample['total_label']).type(torch.FloatTensor).to('cuda')
                all_audio = audio_old.unsqueeze(0)
                all_video = video_old.unsqueeze(0)
                all_video_st = video_st_old.unsqueeze(0)
                all_total_label = total_label_old.unsqueeze(0)
                Flag  =  False
            else:
                current_each_sample = current_sample[i]
                audio_old, video_old, video_st_old, total_label_old = torch.from_numpy(current_each_sample['audio']).to('cuda'), torch.from_numpy(current_each_sample['video_s']).to('cuda'), torch.from_numpy(current_each_sample['video_st']).to('cuda'), torch.tensor(current_each_sample['total_label']).type(torch.FloatTensor).to('cuda')
                all_audio = torch.cat((all_audio, audio_old.unsqueeze(0)),dim=0)
                all_video = torch.cat((all_video, video_old.unsqueeze(0)),dim=0)
                all_video_st = torch.cat((all_video_st, video_st_old.unsqueeze(0)),dim=0)
                all_total_label = torch.cat((all_total_label, total_label_old.unsqueeze(0)),dim=0)

        
        all_output, a_prob, v_prob, _ = model(all_audio, all_video, all_video_st)
        all_class_loss = criterion(all_output, all_total_label.long())
        # print("old data loss is {}".format(all_class_loss))
        loss += all_class_loss
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(audio), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader), loss.item()))

def test(model: nn.Module, data_loader: torch.utils.data.DataLoader):
    model.eval()
    correct = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        output = model(input)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
    return correct / len(data_loader.dataset)
