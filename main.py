import argparse

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers
# from chainer.utils import type_check
from updater import SketchUpdater
from sketch_dataset import SketchDataset
from sketch_visualizer import out_image

from net import VGG,Generator,Discriminator


def main():
    parser = argparse.ArgumentParser(description='chainer implementation of sketch simplification')
    parser.add_argument('--batchsize', '-b', type=int, default=8,
                        help='Number of images in each mini-batch')
    parser.add_argument('--cropsize', '-c', type=int, default=384,
                        help='Crop size of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    # that is, to go through the whole dataset for 100 times
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='./dataset/sketch',
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default='out_models',
                        help='Directory to output the result')
    parser.add_argument('--model', '-m', default='./out_models/G',
                        help='Resume the weights from model')
    parser.add_argument('--dis', '-d', default='./out_models/D',
                        help='Resume the weights from discriminator')
    parser.add_argument('--resume', '-r', default='./out_models/R',
                        help='Resume the training from snapshot')
    parser.add_argument('--snapshot_interval', type=int, default=500,
                        help='Interval of snapshot')# for each 500 iterations, save the model
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')# each epoch contains 100 iterations
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    vgg = VGG()
    # serializers.load_npz('models/nopre-dec-7000.npz', vgg)
    serializers.load_npz('models/fnft-dec-5000.npz', vgg)
    # serializers.load_npz('models/vgg16.model', vgg)
    model = Generator()
    dis = Discriminator()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU
        vgg.to_gpu()
        dis.to_gpu()

    '''vgg has been trained, so we just need to initialize two optimizers for Discriminator and Generator'''
    # Setup an optimizer
    opt_model = chainer.optimizers.Adam(alpha=0.0002, beta1=0.9)#, weight_decay_rate=0.001)
    opt_model.setup(model)# link
    
    # opt_model.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
    opt_dis = chainer.optimizers.Adam(alpha=0.0001, beta1=0.5)#, weight_decay_rate=0.001)
    opt_dis.setup(dis)
    # opt_dis.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')

    train_d = SketchDataset(args.dataset, size=args.cropsize)
    test_d = SketchDataset(args.dataset, size=args.cropsize, train=False)
    
    train_iter = chainer.iterators.SerialIterator(train_d, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_d, args.batchsize)

    # Set up a trainer
    updater = SketchUpdater(
        models=(model, vgg, dis),
        iterator={
            'main': train_iter,
            'test': test_iter},
        optimizer={
            'model': opt_model,
            'dis': opt_dis
        },
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')# 500
    display_interval = (args.display_interval, 'iteration')  # 100
    
    trainer.extend(extensions.snapshot(
        filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'model/loss', 'dis/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=100))
    trainer.extend(
        out_image(
            updater, model, vgg, dis,
            2,2, args.out, args.cropsize),
        trigger=snapshot_interval)

    def reload_data(trainer):
        trainer.updater.get_iterator('main').dataset.reloadimgs()
        trainer.updater.get_iterator('test').dataset.reloadimgs()

    trainer.extend(reload_data, trigger=(5, 'epoch'))

    if args.model:
        chainer.serializers.load_npz(args.model, model)
    if args.dis:
        chainer.serializers.load_npz(args.dis, dis)
    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
