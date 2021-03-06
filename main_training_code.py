import os
import time
import csv
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import models
from loss_function import *
from dataloaders.nyu import NYUDataset
from dataloaders.kitti import KITTIDataset
import configuration_file
cudnn.benchmark = True
import models
from metrics import AverageMeter, Result
import utils

args = utils.parse_command()
print(args)
if configuration_file.GPU == True:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Set the GPU.
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Set the CPU
fieldnames = ['rmse', 'mae', 'delta1', 'absrel',
              'lg10', 'mse', 'delta2', 'delta3', 'data_time', 'gpu_time']
best_fieldnames = ['best_epoch'] + fieldnames
best_result = Result()
best_result.set_to_worst()


# load dataset and split into train and val
def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")
    datasets = configuration_file.datasets_path  # location of the dataset
    traindir = os.path.join(datasets, args.data, 'train')
    valdir = os.path.join(datasets, args.data, 'val')
    train_loader = None
    val_loader = None
    # load nyudepthv2 dataset
    if args.data == 'nyudepthv2':
        if not args.evaluate:  # load training data
            train_dataset = NYUDataset(traindir, split='train',
                                       modality=args.modality)
        # load validation data
        val_dataset = NYUDataset(valdir, split='val',
                                 modality=args.modality)
    # load kitti datasets
    elif args.data == 'kitti':
        if not args.evaluate:  # load training data
            train_dataset = KITTIDataset(traindir, type='train',
                                         modality=args.modality)
        # load validation data
        val_dataset = KITTIDataset(valdir, type='val',
                                   modality=args.modality)

    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of nyudepthv2 or kitti.')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    # put construction of train loader here, for those who are interested in testing only
    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id: np.random.seed(work_id))
        # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created...")

    return train_loader, val_loader


def main():
    global args, best_result, output_directory, train_csv, test_csv
    print(args)
    start = 0


    # evaluation mode
    if args.evaluate:
        datasets = configuration_file.datasets_path
        valdir = os.path.join(datasets, args.data, 'val')
        print("Validation directory ", valdir)
        if args.data == 'nyudepthv2':
            from dataloaders.nyu import NYUDataset
            val_dataset = NYUDataset(valdir, split='val', modality=args.modality)
        else:
            raise RuntimeError('Dataset not found.')

        #set batch size to be 1 for validation
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
        print("=> validation loaders created.")
        assert os.path.isfile(args.evaluate), \
            "=> no model found at '{}'".format(args.evaluate)
        print("=> loading model '{}'".format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        if type(checkpoint) is dict:
            args.start_epoch = checkpoint['epoch']
            best_result = checkpoint['best_result']
            model = checkpoint['model']
            print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        else:
            model = checkpoint
            args.start_epoch = 0
        output_directory = os.path.dirname(args.evaluate)
        validate(val_loader, model, args.start_epoch, write_to_file=False)

        return




    # resume from a particular check point
    elif args.resume:
        chkpt_path = args.resume
        assert os.path.isfile(chkpt_path), "=> no checkpoint found at '{}'".format(chkpt_path)
        print("=> loading checkpoint '{}'".format(chkpt_path))
        checkpoint = torch.load(chkpt_path)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch'] + 1 # load epoch number
        start = start_epoch # resume from the checkpoint epoch
        best_result = checkpoint['best_result'] # load best result
        model = checkpoint['model'] # load model
        optimizer = checkpoint['optimizer'] # load optimizer
        output_directory = os.path.dirname(os.path.abspath(chkpt_path))
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        train_loader, val_loader = create_data_loaders(args) # create data loader
        args.resume = True



    # create new model if checkpoint does not exist
    elif args.train:
        train_loader, val_loader = create_data_loaders(args) # load train and validation data
        print("=> creating Model ({}-{}) ...".format(args.arch, args.decoder))
        in_channels = len(args.modality)
        if args.arch == 'MobileNet': # if encoder is MobileNet
            model = models.MobileNetSkipAdd(output_size=train_loader.dataset.output_size) # MobileNet model is created
        else:
            model = models.MobileNetSkipAdd(output_size=train_loader.dataset.output_size)  # by default MobileNet

        print("=> model created.")
        optimizer = torch.optim.SGD(model.parameters(), args.lr, \
                                    momentum=args.momentum, weight_decay=args.weight_decay) # configure optimizer

        if configuration_file.GPU == True:
            if configuration_file.MULTI_GPU == True:  # training on multiple GPU
                model = torch.nn.DataParallel(model).cuda()
            else:  # training on single GPU
                model = model.cuda()
        else:
            pass

    # define loss function  and optimizer
    if args.criterion == 'l2':
        if configuration_file.GPU == True:
            criterion = MaskedMSELoss().cuda()
        else:
            criterion = MaskedMSELoss()
    elif args.criterion == 'l1':
        if configuration_file.GPU == True:
            criterion = MaskedL1Loss().cuda()
        else:
            criterion = MaskedL1Loss()

    # create results folder, if not already exists
    output_directory = utils.get_output_directory(args)

    if not os.path.exists(output_directory):  # create new directory
        os.makedirs(output_directory)
    train_csv = os.path.join(output_directory, 'train.csv')  # store training result
    test_csv = os.path.join(output_directory, 'test.csv')  # store test result
    best_txt = os.path.join(output_directory, 'best.txt')  # store best result

    # create new csv files with only header
    if not args.resume:
        with open(train_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        with open(test_csv, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # training is strarted from here
    for epoch in range(start, args.epochs):
        utils.adjust_learning_rate(optimizer, epoch, args.lr)
        train(train_loader, model, criterion, optimizer, epoch)  # train for one epoch
        result, img_merge = validate(val_loader, model, epoch)  # evaluate on validation set

        # remember best rmse and save checkpoint
        is_best = result.rmse < best_result.rmse # compare result of the current epoch and best result
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write(
                    "epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                        format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae,
                               result.delta1,
                               result.gpu_time))
            if img_merge is not None:
                img_filename = output_directory + '/comparison_best.png'
                utils.save_image(img_merge, img_filename)

        utils.save_checkpoint({
            'args': args,
            'epoch': epoch,
            'arch': args.arch,
            'model': model,
            'best_result': best_result,
            'optimizer': optimizer,
        }, is_best, epoch, output_directory)


# execute for each epoch
def train(train_loader, model, criterion, optimizer, epoch):
    average_meter = AverageMeter()
    model.train()  # switch to train mode
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        if configuration_file.GPU == True:
            input, target = input.cuda(), target.cuda()
            torch.cuda.synchronize()
        else:
            pass
        data_time = time.time() - end

        # compute pred
        end = time.time()
        pred = model(input)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()  # compute gradient and do SGD step
        optimizer.step()
        if configuration_file.GPU == True:
            torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                epoch, i + 1, len(train_loader), data_time=data_time,
                gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                         'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                         'gpu_time': avg.gpu_time, 'data_time': avg.data_time}) # store result of every epoch


# Validation
def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval()  # switch to evaluate mode
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if configuration_file.GPU == True:
            input, target = input.cuda(), target.cuda()
        # torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        # torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        skip = 50

        if args.modality == 'rgb':
            rgb = input

        if i == 0:
            img_merge = utils.merge_into_row(rgb, target, pred)
        elif (i < 8 * skip) and (i % skip == 0):
            row = utils.merge_into_row(rgb, target, pred)
            img_merge = utils.add_row(img_merge, row)
        elif i == 8 * skip:
            filename = output_directory + '/comparison_' + str(epoch) + '.png'
            utils.save_image(img_merge, filename)

        if (i + 1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                i + 1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
          'RMSE={average.rmse:.3f}\n'
          'MAE={average.mae:.3f}\n'
          'Delta1={average.delta1:.3f}\n'
          'REL={average.absrel:.3f}\n'
          'Lg10={average.lg10:.3f}\n'
          't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                             'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                             'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    return avg, img_merge


if __name__ == '__main__':
    main()
