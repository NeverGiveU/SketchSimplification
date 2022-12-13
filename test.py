import argparse

import chainer
from chainer import cuda, serializers, Variable
from net import Generator
from PIL import Image, ImageOps
import numpy as np
import os

def read_image(filepath,model,args,out_fn):
    if os.path.isfile(filepath):
        name = filepath.split('/')[-1]
        img,size = load_image(filepath, args.scale)
        # save_image(img, os.path.join(args.out_fn, 'z-'+name),img.shape)
        gen = simplify(img, model, args)
        save_image(gen, os.path.join(out_fn, name), size)
    elif os.path.exists(filepath):
        fs = os.listdir(filepath)
       
        print('total imgs: {}'.format(len(fs)))
        i = 0
        for scale in [0.50, 0.75, 1.0, 1.25]:
            if not os.path.exists(os.path.join(out_fn, str(scale))):
                os.mkdir(os.path.join(out_fn, str(scale)))
            for (kk, fn) in enumerate(fs):
                
                if (fn.split('.')[-1] == 'bmp') or (fn.split('.')[-1] == 'jpeg') or (fn.split('.')[-1] == 'jpg') or (fn.split('.')[-1] == 'png'):
                    
                    
                    i += 1
                    if i % 100 == 0:
                        print('finished %d imgs'%i)
    
                    data = Image.open(os.path.join(filepath,fn))
                    data = data.resize((int(scale * data.size[0]), int(scale * data.size[1])), Image.ANTIALIAS)
                    # if data.size[0] * data.size[1] >= 250000:
                    #     data = data.resize((int(250000 / data.size[1]), int(250000 / data.size[0])), Image.ANTIALIAS)
                    img, size = load_image(os.path.join(filepath, fn), data.size)
                    gen = simplify(img, model, args)
                    save_image(gen, os.path.join(out_fn, str(scale) , fn),  size)
                else:
                    continue

def load_image(fn, size):
    img = ImageOps.autocontrast(Image.open(fn).convert('L'), 0)
    # print(size)
    try:
        img = img.resize((size[0], size[1]))
    except:
        pass
    # scale = max(img.size[0], img.size[1]) / 512.0
    # if scale > 1:
    #     img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)))
    # img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)))
    img = np.asarray(img, dtype=np.float32)
    img2 = np.pad(img, ((31, 31), (31, 31)), 'edge')#'constant', constant_values=255)
    img2 = img2[np.newaxis, np.newaxis, :, :] / 127.5 - 1
    return img2, img.shape

def save_image(im, outfn, shape):
    img = (im[0][0] + 1) * 127.5
    img = np.uint8(img[31:shape[0]+31,31:shape[1]+31])
    img = Image.fromarray(img)
    img.save(outfn)

def simplify(img,model,args):
    if args.gpu >= 0:
        img = cuda.to_gpu(img)
    img = Variable(img)

    with chainer.no_backprop_mode():
        with chainer.using_config('train', False):
            gen = model(img)
    gen = gen.data
    if args.gpu >= 0:
        gen = cuda.to_cpu(gen)
    # save_image(gen, os.path.join(args.out_fn, '{}-{}'.format(args.scale,name)), img.shape)
    return gen

model = Generator()

parser = argparse.ArgumentParser(description='Demo of sketch simplification')
parser.add_argument('--img_fn', '-i', type=str, default='test_2/input',
                    help='Directory of the rough sketch')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--scale', '-s', type=float, default=1,
                    help='Scale of the rough sketch')
parser.add_argument('--out_fn', '-o', type=str, default='test_2/output',
                    help='Directory to output the result')
parser.add_argument('--model', '-m', type=str, default='pre-fa',
                    help='Directory of the trained model')
args = parser.parse_args()

    
if args.model != None:
    # serializers.load_npz(modelpath, model)
    serializers.load_npz('./models/model_iter_39000.npz', model)
    # serializers.load_npz(args.model, model)

if args.gpu >= 0:
    print('use cuda? Yes')
    cuda.get_device(args.gpu).use()  # Make a specified GPU current
    model.to_gpu()  # Copy the model to the GPU
else:
    print('use cuda? No')

try:
    # os.mkdir(args.out_fn)
    os.mkdir(os.path.join(args.out_fn))
except:
    pass
    
read_image(args.img_fn,model, args,os.path.join(args.out_fn))
