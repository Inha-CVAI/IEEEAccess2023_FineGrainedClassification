from __future__ import print_function
from card import CarDataset
from tqdm import tqdm
from distutils.log import error
import os
import torch
import argparse
from dataset import *
from utils import *
import warnings
warnings.filterwarnings(action='ignore')
from typing import OrderedDict
import csv


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
transform = {
    'train': transforms.Compose([transforms.Resize(550),
                                 transforms.CenterCrop(448),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                                 ]),
    'test': transforms.Compose([transforms.Resize(448),
                                transforms.CenterCrop(448),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
}


def main():
    parser = argparse.ArgumentParser('FGVC', add_help=False)
    parser.add_argument('--epochs', type=int, default=300, help="training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size for training")
    parser.add_argument('--load', type=str, default="cub_resnet50_4", help="load from saved model path")
    parser.add_argument('--dataset_name', '-d', type=str, default="cub", choices=['cub', 'car', 'air'],
                        help="dataset name")
    parser.add_argument('--topn', type=int, default=4, help="parts number")
    parser.add_argument('--backbone', type=str, default="resnet50", help="backbone")
    parser.add_argument('--lr', type=float, default=2e-3, help="learning rate")
    args, _ = parser.parse_known_args()
    epochs = args.epochs
    batch_size = args.batch_size


    ## Data
    data_config = {"air": [100, "../data/fgvc-aircraft-2013b"],
                   "car": [196, "../data/stanford_cars"],
                   "cub": [200, "../data/CUB_200_2011"],
                   }
    dataset_name = args.dataset_name
    classes_num, data_root = data_config[dataset_name]
    # print('Dataset: {}'.format(dataset_name.upper()))
    # print('Dataset: {} | labels num: {} | data path: {}'.format(dataset_name.upper(), classes_num, data_root))
    if dataset_name == 'air':
        trainset = AIR(root=data_root, is_train=True, data_len=None)
        testset = AIR(root=data_root, is_train=False, data_len=None)
    elif dataset_name == 'car':
        trainset = CarDataset(root=data_root, phase='train',
                              resize=(448, 448))  # CAR(root=data_root, is_train=True, data_len=None)
        testset = CarDataset(root=data_root, phase='val',
                             resize=(448, 448))  # CAR(root=data_root, is_train=False, data_len=None)
    elif dataset_name == 'cub':
        trainset = CUB(root=data_root, is_train=True, data_len=None)
        testset = CUB(root=data_root, is_train=False, data_len=None)
    else:
        raise Exception('Dataset must be in [air, car, cub]')
    num_workers = 16 if torch.cuda.is_available() else 0

    topn = args.topn

    ## Model
    net = load_model(backbone=args.backbone, pretrain=True, require_grad=False, classes_num=classes_num, topn=topn)

    state_dict = dict()
    for i in range(3):
        pth_file = os.path.join(args.load, 'model' + str(i) + '.pth')
        state_dict_part = torch.load(pth_file)
        state_dict_part_dict = dict(state_dict_part)
        state_dict.update(state_dict_part_dict)
    state_dict = OrderedDict(state_dict)
    net.load_state_dict(state_dict)

    # print('Load model from ' + args.load)
    # print(net)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        net = net.to(device)
    else:
        device = torch.device('cpu')

    # CELoss = nn.CrossEntropyLoss()
    # mse_loss = nn.MSELoss()

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, drop_last=False)

    results = []

    net.eval()
    num_correct = [0] * 5
    with torch.no_grad():
        for idx, (inputs, targets, img_path) in enumerate(testloader):
            img_names = [path.split('/')[-1] for path in img_path]

            if torch.cuda.is_available():
                inputs, targets = inputs.to(device), targets.to(device)
            y1, y2, y3, y4, yp1, yp2, yp3, yp4, top_n_prob, f1_m, f1, f2_m, f2, f3_m, f3, out = net(inputs,
                                                                                                    is_train=False)

            _, p1 = torch.max(y1.data, 1)
            _, p2 = torch.max(y2.data, 1)
            _, p3 = torch.max(y3.data, 1)
            _, p4 = torch.max(y4.data, 1)
            _, p5 = torch.max((y1 + y2 + y3 + y4).data, 1)

            num_correct[0] += p1.eq(targets.data).cpu().sum()
            num_correct[1] += p2.eq(targets.data).cpu().sum()
            num_correct[2] += p3.eq(targets.data).cpu().sum()
            num_correct[3] += p4.eq(targets.data).cpu().sum()
            num_correct[4] += p5.eq(targets.data).cpu().sum()

            for i in range(inputs.size(0)):
                results.append({'Image name': img_names[i], 'Predict class': p5[i].item()})

    total = len(testset)
    acc1 = 100. * float(num_correct[0]) / total
    acc2 = 100. * float(num_correct[1]) / total
    acc3 = 100. * float(num_correct[2]) / total
    acc4 = 100. * float(num_correct[3]) / total
    acc_test = 100. * float(num_correct[4]) / total

    # result_str = 'acc1 = %.5f | acc2 = %.5f | acc3 = %.5f | acc4 = %.5f | acc_test = %.5f \n' % (
    #     acc1, acc2, acc3, acc4, acc_test)
    # print(result_str)
    # with open(exp_dir + '/results_test_all.txt', 'a') as file:
    #     file.write(result_str)
    print('Dataset: {}, Accuracy: {:.2f}\n'.format(dataset_name.upper(), acc_test))

    with open('/results/' + dataset_name + '_results.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Image name', 'Predict class'])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
