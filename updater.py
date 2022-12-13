#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np


def gram_matrix(y):
    b, ch, h, w = y.data.shape
    features = F.reshape(y, (b, ch, w*h))
    gram = F.batch_matmul(features, features, transb=True)/np.float32(ch*w*h)
    return gram

class SketchUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.model, self.vgg, self.dis = kwargs.pop('models')
        # self.model, self.vgg = kwargs.pop('models')
        # self.mid = None
        # self.position = 0
        super(SketchUpdater, self).__init__(*args, **kwargs)

    def loss_dis(self, dis, pred_f, pred_r, pred_f2=0.0,lam1=1e-2):
        # batchsize = len(pred_f)
        L1 = lam1*F.mean(F.softplus(-pred_r)) #/ batchsize
        L2 = lam1*F.mean(F.softplus(pred_f)) #/ batchsize
        # L2 += F.mean(F.softplus(pred_f2)) #/ batchsize
        # L1 = F.mean((pred_r - 1.0) ** 2)2
        # L2 = F.mean(pred_f ** 2)
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def loss_model2(self, model, feat_f=0, feat_r=0):
        # batchsize = len(pred_f)
        L2 = F.mean_squared_error(feat_f, feat_r)  # / batchsize
        loss = L2
        chainer.report({'loss': loss}, model)
        return loss

    def loss_model(self, model, pred_f, feat_f=0, feat_r=0, weight_f=0.0, weight_r=0.0, lam1=1e-3, lam2=1e-3):
        # batchsize = len(pred_f)
        # L1 = F.mean((pred_f - 1.0) ** 2)
        L1 = F.mean(F.softplus(-pred_f))#/ batchsize
        # L1 += F.mean(F.softplus(-pred_f2))#/ batchsize
        # L3 = F.mean_squared_error(seg_f, seg_r)
        L2 = 0
        # L2 = F.mean_squared_error(F.normalize(feat_f), F.normalize(feat_r))  # / batchsize
        # L2 = F.mean_squared_error(feat_f, feat_r)  # / batchsize
        # k = 0
        # weights = self.dis.lin.W
        # for (y_fake,y_real,y_fake2) in zip(feat_f,feat_r,feat_f2):
        for (y_fake,y_real) in zip(feat_f,feat_r):
            # gm_f = gram_matrix(y_fake)
            # gm_r = gram_matrix(y_real)
            # L2 += F.mean_squared_error(gm_f,gm_r)
            # weights_feat = self.dis.convs[k].c1.W.data
            # L2 += F.mean_squared_error(F.normalize(y_fake)*weights_feat, F.normalize(y_real)*weights_feat)# / batchsize
            # L2 += weights[0][k]*F.mean_squared_error(y_fake, y_real)# / batchsize
            # k=k+1
            # L2 += F.mean_squared_error(F.normalize(y_fake), F.normalize(y_real))# / batchsize
            L2 += F.mean_squared_error(y_fake, y_real)# / batchsize
        loss = lam1*L1 + lam2*L2# + lam2*L311
        # loss = L2
        chainer.report({'loss': loss}, model)
        return loss

    def update_core(self):
        model_optimizer = self.get_optimizer('model')
        dis_optimizer = self.get_optimizer('dis')
        # vgg_optimizer = self.get_optimizer('vgg')
        
        model, vgg, dis = self.model, self.vgg, self.dis
        # model, vgg = self.model, self.vgg
        xp = model.xp
        # iterk = 2
        #
        # for i in range(iterk):
        #     batch = self.get_iterator('main').next()
        #     batchsize = len(batch)
        #     w_in = batch[0][0].shape[0]
        #     w_out = batch[0][1].shape[0]
        #
        #     x_in = xp.zeros((batchsize, 1, w_in, w_in)).astype("f")
        #     t_out = xp.zeros((batchsize, 1, w_out, w_out)).astype("f")
        #
        #     for i in range(batchsize):
        #         x_in[i,0,:] = xp.asarray(batch[i][0])
        #         t_out[i,0,:] = xp.asarray(batch[i][1])
        #
        #     x_in = Variable(x_in)
        #     t_out = Variable(t_out)
        #
        #     with chainer.using_config('train', True):
        #         x_out = model(x_in)
        #     #print(F.mean_squared_error(x_out, t_out))
        #     with chainer.using_config('train', False):
        #         y_fake = vgg(x_out)
        #         y_real = vgg(t_out)
        #         y_in = vgg(x_in)
        #
        #     with chainer.using_config('train', True):
        #         pred_f,diff_f = dis(y_fake,y_in)
        #         pred_r,diff_r = dis(y_real,y_in)
        #
        #     # loss = self.loss()
        #     # vgg_optimizer.update(self.loss_vgg, vgg, pred_f, pred_r)
        #     dis_optimizer.update(self.loss_dis, dis, pred_f, pred_r)
        #     # model_optimizer.update(self.loss_model, model, pred_f,diff_f)
        #     # vgg_optimizer.update()
        #     x_in.unchain_backward()
        #     x_out.unchain_backward()

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        w_in = batch[0][0].shape[0]
        w_out = batch[0][1].shape[0]

        x_in = xp.zeros((batchsize, 1, w_in, w_in)).astype("f")
        t_out = xp.zeros((batchsize, 1, w_out, w_out)).astype("f")
        # t_seg = xp.zeros((batchsize, 1, w_out, w_out)).astype("f")
        # rate = 0.0001
        # remembernum = 64
        # position = self.position
        # rem_out = xp.zeros((batchsize, 1, w_out, w_out)).astype("f")
        # if self.mid == None:
        #     mid = xp.zeros((remembernum, 1, w_out, w_out)).astype("f")
        # else:
        #     mid = self.mid

        for i in range(batchsize):
            x_in[i, 0, :] = xp.asarray(batch[i][0])
            t_out[i, 0, :] = xp.asarray(batch[i][1])
            # t_seg[i,0,:] = xp.asarray(batch[i][2])

        x_in = Variable(x_in)
        t_out = Variable(t_out)

        with chainer.using_config('train', True):
            x_out = model(x_in)
            # x_out, x_seg = model(x_in)
            # x_out2 = model(x_in+x_out)

        # if position==remembernum and np.random.random()<rate:
        #     mid[int(np.random.random()*remembernum),0] = x_out.data[int(np.random.random()*batchsize),0]
        # elif position<remembernum:
        #     mid[position,0] = x_out.data[int(np.random.random()*batchsize),0]
        #     position += 1
        # rem_out[:] = mid[np.random.random()]
        # self.mid = mid

        with chainer.using_config('train', False):
            y_fake = vgg(x_out)
            y_real = vgg(t_out)
            # s_fake = vgg(x_seg)
            # s_real = vgg(t_seg)
            # y_in = vgg(x_in)
            # y_fake2 = vgg(x_out2)

        with chainer.using_config('train', True):
            # pred_f,feat_f = dis(y_fake,y_in)
            # pred_r,feat_r = dis(y_real,y_in)
            # pred_f = dis(y_fake)
            # pred_r = dis(y_real)
            pred_f,feat_f = dis(y_fake)
            pred_r,feat_r = dis(y_real)
            # pred_f,feat_f,atten_f = dis(y_fake)
            # pred_r,feat_r,atten_r = dis(y_real)
            # pred_f2,feat_f2 = dis2(s_fake)
            # pred_r2,feat_r2 = dis2(s_real)
        #     # pred_f2,feat_f2 = dis(y_fake2)

        # loss = self.loss()
        # vgg_optimizer.update(self.loss_vgg, vgg, y_fake, y_real)
        # model_optimizer.update(self.loss_model, model, y_fake)
        # vgg_optimizer.update(self.loss_vgg, vgg, pred_f, pred_r)
        # dis_optimizer.update(self.loss_dis, dis, pred_f, pred_r)
        # model_optimizer.update(self.loss_model, model, pred_f, feat_f, feat_r)
        # model_optimizer.update(self.loss_model2, model, y_fake, y_real)
        dis_optimizer.update(self.loss_dis, dis, pred_f, pred_r)
        model_optimizer.update(self.loss_model, model, pred_f, feat_f, feat_r)
        # dis_optimizer.update(self.loss_dis, dis, pred_f, pred_r)
        # model_optimizer.update(self.loss_model, model, pred_f, feat_f, feat_r, x_seg, t_seg)
        # dis_optimizer.update(self.loss_dis, dis, pred_f, pred_r, pred_f2)
        # model_optimizer.update(self.loss_model, model, pred_f, feat_f, feat_r,pred_f2, feat_f2)
        # vgg_optimizer.update()
        x_in.unchain_backward()
        x_out.unchain_backward()


