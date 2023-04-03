from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import *
from nets.net_audiovisual import MMIL_Net
from utils.eval_metrics import segment_level, event_level
import pandas as pd
from torch.backends import cudnn
import random
from util_new_oldsample import EWC, ewc_train,  normal_train, test
import sys
import pdb
def train(args, model, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        audio, video, video_st, total_label, audio_label, visual_label = sample['audio'].to('cuda'), sample['video_s'].to('cuda'), sample['video_st'].to('cuda'), sample['total_label'].type(torch.FloatTensor).to('cuda'), sample['audio_label'].type(torch.FloatTensor).to('cuda'), sample['visual_label'].type(torch.FloatTensor).to('cuda')  #[16,10,128] [16,80,2048], [16,10,512]

        optimizer.zero_grad()
        output, a_prob, v_prob, _ = model(audio, video, video_st)
        # output.clamp_(min=1e-7, max=1 - 1e-7)
        # a_prob.clamp_(min=1e-7, max=1 - 1e-7)
        # v_prob.clamp_(min=1e-7, max=1 - 1e-7)

        # # label smoothing
        # a = 1.0
        # v = 0.9 
        # Pa = a * target + (1 - a) * 0.5
        # Pv = v * target + (1 - v) * 0.5

        # individual guided learning
        # loss = criterion(output, total_label.long()) 
        loss = criterion(output, total_label.long())
        # print(loss)

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval(model, val_loader, set, mode, epoch):
    categories = ['positive', 'negative']
    model.eval()

    # load annotations
    df = pd.read_csv(set, header=0, sep='\t')
    # df_a = pd.read_csv("data/AVVP_eval_audio.csv", header=0, sep='\t')
    # df_v = pd.read_csv("data/AVVP_eval_visual.csv", header=0, sep='\t')

    id_to_idx = {id: index for index, id in enumerate(categories)}
    F_seg_a = []
    F_seg_v = []
    F_seg = []
    F_seg_av = []
    F_event_a = []
    F_event_v = []
    F_event = []
    F_event_av = []

    with torch.no_grad():
        num_all = 0
        num_all_true = 0
        num_all_true_a = 0
        num_all_true_v = 0
        for batch_idx, sample in enumerate(val_loader):
            audio, video, video_st, totol_label, audio_label, visual_label = sample['audio'].to('cuda'), sample['video_s'].to('cuda'),sample['video_st'].to('cuda'), sample['total_label'].to('cuda'), sample['audio_label'].to('cuda'), sample['visual_label'].to('cuda')
            output, a_prob, v_prob, frame_prob = model(audio, video, video_st)
            pred_label = torch.argmax(output,dim=1)
            pred_label_a = torch.argmax(a_prob,dim=1)
            pred_label_v = torch.argmax(v_prob,dim=1)
            num_true = torch.sum(pred_label == totol_label)
            num_true_a = torch.sum(pred_label_a == audio_label)
            num_true_v = torch.sum(pred_label_v == visual_label)
            num_all_true = num_all_true + num_true
            num_all_true_a = num_all_true_a + num_true_a
            num_all_true_v = num_all_true_v + num_true_v
            num = len(totol_label)
            num_all = num + num_all
        accuracy = num_all_true/num_all * 100
        accuracy_a = num_all_true_a/num_all * 100
        accuracy_v = num_all_true_v/num_all * 100
        # if epoch == 10:
        print("Train Epoch: {} total accuracy is {:.2f}%".format(num, accuracy))
        # print("audio accuracy is {:.2f}%".format(accuracy_a))
        # print("visual accuracy is {:.2f}%".format(accuracy_v))
            
    return accuracy


def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Video Parsing')
    parser.add_argument(
        "--dataset", type=str, default='earthquake', help="earthquake or AVE")
    parser.add_argument(
        "--audio_dir", type=str, default='/data/yss/dataset/AVDPR/compress/vggish/', help="audio dir")
    parser.add_argument(
        "--video_dir", type=str, default='/data/yss/dataset/AVDPR/compress/res152_train_256/',
        help="video dir")
    parser.add_argument(
        "--st_dir", type=str, default='/data/yss/dataset/AVDPR/compress/r2plus1d_18/',
        help="video dir")
    parser.add_argument(
        "--label_train", type=str, default="data/AVVP_train_shuffle_split5.csv", help="weak train csv file")
    parser.add_argument(
        "--label_val", type=str, default="data/AVVP_val_pd.csv", help="weak val csv file")
    parser.add_argument(
        "--label_test", type=str, default="data/AVVP_test_pd.csv", help="weak test csv file")
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--K', type=int, default=5, metavar='N',
                        help='task split')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--store_num', type=int, default=100)
    parser.add_argument('--importance', type=float, default=1000, help='importance (default: 1000)')
    parser.add_argument(
        "--model", type=str, default='MMIL_Net', help="with model to use")
    parser.add_argument(
        "--mode", type=str, default='test', help="with mode to use")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument(
        "--model_save_dir", type=str, default='models/', help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='IAVF',
        help="save model name")
    parser.add_argument(
        '--gpu', type=str, default='5', help='gpu device number')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.dataset == 'earthquake':
        args.audio_dir = '/data/yss/dataset/AVDPR/compress/vggish/'
        args.video_dir = '/data/yss/dataset/AVDPR/compress/res152_train_256/'
        args.st_dir = '/data/yss/dataset/AVDPR/compress/r2plus1d_18/'
        args.label_train = 'data/AVVP_train_shuffle_split5.csv'
        args.label_val = 'data/AVVP_val_pd.csv'
        args.label_test = 'data/AVVP_test_pd.csv'
    else:
        args.audio_dir = '/data/yss/dataset/AVE_Dataset/AVE_compress/vggish_test/' 
        args.video_dir = '/data/yss/dataset/AVE_Dataset/AVE_compress/res152_test_256/' 
        args.st_dir = '/data/yss/dataset/AVE_Dataset/AVE_compress/r2plus1d_18_test/' 
        args.label_train = '/data/yss/dataset/AVE_Dataset/AVE_train.csv'
        args.label_val = '/data/yss/dataset/AVE_Dataset/AVE_test.csv'
        args.label_test = '/data/yss/dataset/AVE_Dataset/AVE_test.csv'
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

    if args.model == 'MMIL_Net':
        model = MMIL_Net().to('cuda')
    else:
        raise ('not recognized')

    save_checkpoint = '{}-{}'.format(args.checkpoint, args.K)
    
    log_file = './logs/IAVF-K{}_{}.txt'.format(args.K, args.dataset)
    
    temp = sys.stdout
    f = open(log_file, 'w')
    sys.stdout = f
    for num in range(args.K):
        print("split the dataset to {} block".format(args.K))
        print("current is {} block".format(num))
        if args.mode == 'train':
            # pdb.set_trace()
            train_dataset = LLP_dataset(label=args.label_train, dataset = args.dataset,audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform = transforms.Compose([
                                                ToTensor()]), flag = "train", K=args.K, num_now=num)
            print("the len of current dataset is {}".format(len(train_dataset)))
            val_dataset = LLP_dataset(label=args.label_val, dataset = args.dataset,audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform = transforms.Compose([
                                                ToTensor()]))
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory = True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory = True)
            val_loader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory = True)

            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            # criterion = nn.BCELoss()
            criterion = nn.CrossEntropyLoss()
            if num == 0:
                best_F = 0
                for epoch in range(1, args.epochs + 1):
                    print("for training data")
                    F = eval(model, val_loader_train, args.label_val,args.mode, epoch)
                    train(args, model, train_loader, optimizer, criterion, epoch=epoch)
                    scheduler.step(epoch)
                    print("for training data")
                    F = eval(model, val_loader_train, args.label_val,args.mode, epoch)
                    print("for testing data")
                    F = eval(model, val_loader, args.label_val,args.mode, epoch)
                    if F >= best_F:
                        best_F = F
                        torch.save(model.state_dict(), args.model_save_dir + save_checkpoint + ".pt")
            else:
                best_F = 0
                old_val_dataset = LLP_dataset(label=args.label_train, dataset = args.dataset,audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform = transforms.Compose([
                                                ToTensor()]), flag = "train", K=args.K, num_now=num-1)
                old_train_loader = DataLoader(old_val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory = True)
                old_tasks = []
                for sub_task in range(num):
                    old_tasks = old_tasks + LLP_dataset(label=args.label_train, dataset = args.dataset,audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform = transforms.Compose([
                                                ToTensor()]), flag = "train", K=args.K, num_now=sub_task).get_sample(args.store_num)
                # old_tasks = random.sample(old_tasks, k=200)
                for epoch in range(1, args.epochs + 1):
                    ewc_train(args, model, train_loader, optimizer, criterion, epoch=epoch, ewc = EWC(model, old_train_loader), importance = args.importance, old_data = old_tasks)
                    scheduler.step(epoch)
                    print("for training data")
                    F = eval(model, val_loader_train, args.label_val,args.mode, epoch)
                    print("for testing data")
                    F = eval(model, val_loader, args.label_val,args.mode, epoch)
                    if F >= best_F:
                        best_F = F
                        torch.save(model.state_dict(), args.model_save_dir + save_checkpoint + ".pt")
        elif args.mode == 'val':
            test_dataset = LLP_dataset(label=args.label_val, dataset = args.dataset,audio_dir=args.audio_dir, video_dir=args.video_dir,
                                        st_dir=args.st_dir, transform=transforms.Compose([
                    ToTensor()]))
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
            model.load_state_dict(torch.load(args.model_save_dir + save_checkpoint + ".pt"))
            eval(model, test_loader, args.label_val,args.mode, num)
        else:
            test_dataset = LLP_dataset(label=args.label_test, dataset = args.dataset,audio_dir=args.audio_dir, video_dir=args.video_dir,  st_dir=args.st_dir, transform = transforms.Compose([
                                                ToTensor()]))
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
            model.load_state_dict(torch.load(args.model_save_dir + save_checkpoint + ".pt"))
            eval(model, test_loader, args.label_test,args.mode, num)
    
    f.close()
if __name__ == '__main__':
    main()
