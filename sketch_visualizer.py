#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable

def out_image(updater, model, vgg, dis, rows, cols, dst, size):
    @chainer.training.make_extension()
    def make_image(trainer):
        n_images = rows * cols
        xp = model.xp
        
        w_in = size
        w_out = size
        in_ch = 1
        out_ch = 1
        
        in_all = np.zeros((n_images, in_ch, w_in, w_in)).astype("f")
        gt_all = np.zeros((n_images, out_ch, w_out, w_out)).astype("f")
        gen_all = np.zeros((n_images, out_ch, w_out, w_out)).astype("f")
        # atf_all = np.zeros((n_images, 1, w_in//4*5,w_in//4)).astype('f')
        # atr_all = np.zeros((n_images, 1, w_in//4*5,w_in//4)).astype('f')
        # seg_all = np.zeros((n_images, out_ch, w_out, w_out)).astype("f")
        # st_all = np.zeros((n_images, out_ch, w_out, w_out)).astype("f")
        
        for it in range(n_images):
            batch = updater.get_iterator('test').next()
            batchsize = len(batch)

            x_in = xp.zeros((batchsize, in_ch, w_in, w_in)).astype("f")
            t_out = xp.zeros((batchsize, out_ch, w_out, w_out)).astype("f")
            # st = xp.zeros((batchsize, out_ch, w_out, w_out)).astype("f")
            # hist_in = xp.zeros((batchsize, 256)).astype("f")

            for i in range(batchsize):
                # hist,edge = np.histogram(batch[i][0], 256)
                # hist_in[i,:] = xp.asarray(hist.astype("f"))
                x_in[i,:] = xp.asarray(batch[i][0])
                t_out[i,:] = xp.asarray(batch[i][1])
                # st[i,:] = xp.asarray(batch[i][2])

            x_in = Variable(x_in)
            # hist_in = Variable(hist_in)

            with chainer.no_backprop_mode():
                with chainer.using_config('train', False):
                    # x_out = model(x_in,hist_in)
                    x_out = model(x_in)
                    # x_out, seg = model(x_in)
                    # y_fake = vgg(x_out)
                    # y_real = vgg(t_out)
                    # pred_f,feat_f,atten_f = dis(y_fake)
                    # pred_r,feat_r,atten_r = dis(y_real)

            in_all[it,:] = x_in.data.get()[0,:]
            gt_all[it,:] = t_out.get()[0,:]
            gen_all[it,:] = x_out.data.get()[0,:]
            # seg_all[it,:] = seg.data.get()[0,:]
            # st_all[it,:] = st.get()[0,:]
            # atf_all[it,:] = atten_f.data.get()[0,:].reshape((-1,w_in//4))
            # atr_all[it,:] = atten_r.data.get()[0,:].reshape((-1,w_in//4))

        def save_image(x, name, mode=None):
            _, C, H, W = x.shape
            x = x.reshape((rows, cols, C, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            if C==1:
                x = x.reshape((rows*H, cols*W))
            else:
                x = x.reshape((rows*H, cols*W, C))

            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/image_{}_{:0>8}.jpg'.format(name, trainer.updater.iteration)
            # out_models/preview/image_in,gen,gt_xxx.jpg
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(x, mode=mode).convert('L').save(preview_path)

        x = np.asarray(np.clip(in_all * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        save_image(x, "in")

        x = np.asarray(np.clip(gen_all * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        save_image(x, "gen")

        x = np.asarray(np.clip(gt_all * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        save_image(x, "gt")

        # x = np.asarray(np.clip(atf_all * 255.0, 0.0, 255.0), dtype=np.uint8)
        # save_image(x, "atf_all")
        #
        # x = np.asarray(np.clip(atr_all * 255.0, 0.0, 255.0), dtype=np.uint8)
        # save_image(x, "atr_all")
        # x = np.asarray(np.clip(127.5 - gen_all * 127.5, 0.0, 255.0), dtype=np.uint8)
        # save_image(x, "gen")
        #
        # x = np.asarray(np.clip(127.5 - gt_all * 127.5, 0.0, 255.0), dtype=np.uint8)
        # save_image(x, "gt")

        # x = np.asarray(np.clip(seg_all * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        # save_image(x, "seg")
        #
        # x = np.asarray(np.clip(st_all * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
        # save_image(x, "st")
    return make_image
