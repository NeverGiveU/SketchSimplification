import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers

class BottleNeckA(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2):
        super(BottleNeckA, self).__init__(
            conv1 = L.Convolution2D(in_size, in_size, 3, stride, 1, initialW=initializers.HeNormal(), nobias=True),
            bn1 = L.BatchNormalization(in_size),
            conv2 = L.Convolution2D(in_size, out_size, 3, 1, 1, initialW=initializers.HeNormal(), nobias=True),
            bn2 = L.BatchNormalization(out_size),
            conv3 = L.Convolution2D(in_size, out_size, 3, stride, 1, initialW=initializers.HeNormal(), nobias=True),
            bn3 = L.BatchNormalization(out_size)
        )

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = self.bn2(self.conv2(h1))
        h2 = self.bn3(self.conv3(x))
        return F.relu(h1 + h2)

class BottleNeckC(chainer.Chain):

    def __init__(self, in_size, ch, out_size, stride=2):
        super(BottleNeckC, self).__init__(
            conv1 = L.Deconvolution2D(in_size, in_size, 4, stride, 1, initialW=initializers.HeNormal(), nobias=True),
            bn1 = L.BatchNormalization(in_size),
            conv2 = L.Convolution2D(in_size, out_size, 3, 1, 1, initialW=initializers.HeNormal(), nobias=True),
            bn2 = L.BatchNormalization(out_size),
            conv3 = L.Deconvolution2D(in_size, out_size, 4, stride, 1, initialW=initializers.HeNormal(), nobias=True),
            bn3 = L.BatchNormalization(out_size)
        )

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = self.bn2(self.conv2(h1))
        h2 = self.bn3(self.conv3(x))
        return F.relu(h1 + h2)

class BottleNeckB(chainer.Chain):

    def __init__(self, in_size, ch):
        super(BottleNeckB, self).__init__(
            conv1 = L.Convolution2D(in_size, ch, 1, 1, 0, initialW=initializers.HeNormal(), nobias=True),
            bn1 = L.BatchNormalization(ch),
            conv2 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=initializers.HeNormal(), nobias=True),
            bn2 = L.BatchNormalization(ch),
            conv3 = L.Convolution2D(ch, in_size, 1, 1, 0, initialW=initializers.HeNormal(), nobias=True),
            bn3 = L.BatchNormalization(in_size)
        )

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))

        return F.relu(h + x)

class ResidualBlock1(chainer.Chain):
    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(ResidualBlock1, self).__init__()
        self.add_link('a', BottleNeckA(in_size, ch, out_size, stride))
        for i in range(1, layer):
            self.add_link('b{}'.format(i), BottleNeckB(out_size, ch))
        self.layer = layer

    def __call__(self, x):
        h = self.a(x)
        for i in range(1, self.layer):
            h = self['b{}'.format(i)](h)

        return h

class ResidualBlock2(chainer.Chain):
    def __init__(self, layer, in_size, ch, out_size, stride=2):
        super(ResidualBlock2, self).__init__()
        self.add_link('a', BottleNeckC(in_size, ch, out_size, stride))
        for i in range(1, layer):
            self.add_link('b{}'.format(i), BottleNeckB(out_size, ch))
        self.layer = layer

    def __call__(self, x):
        h = self.a(x)
        for i in range(1, self.layer):
            h = self['b{}'.format(i)](h)

        return h

class ConvBlock(chainer.Chain):
    def __init__(self, n_in, n_out, ksize=3, stride=1, pad=0, cfun=L.Convolution2D, rfun=F.relu, nobias=False):
        super(ConvBlock, self).__init__(
            c1=cfun(n_in, n_out, ksize, stride, pad, nobias=nobias,initialW = initializers.HeNormal()),
            b1=L.BatchNormalization(n_out)
        )
        self.rfun = rfun

    def __call__(self, x):
        h = self.rfun(self.b1(self.c1(x)))
        return h

class DilatedConvBlock(chainer.Chain):
    def __init__(self, n_in, n_out, ksize=3, stride=1, pad=0, dilate=2, rfun=F.relu):
        super(DilatedConvBlock, self).__init__(
            c1=L.DilatedConvolution2D(n_in, n_out, ksize, stride, pad, dilate, initialW=initializers.HeNormal()),
            b1=L.BatchNormalization(n_out)
        )
        self.rfun = rfun

    def __call__(self, x):
        h = self.rfun(self.b1(self.c1(x)))
        return h

# class Generator(chainer.Chain):
#     def __init__(self):
#         super(Generator, self).__init__(
#             c0 = ConvBlock(1, 24, ksize=3, stride=1, pad=1),
#             #r0 = ResidualBlock1(1, 1, 24, 24),
#             r1 = ResidualBlock1(1, 24, 24, 48),
#             r2 = ResidualBlock1(1, 48, 48, 96),
#             r3 = ResidualBlock1(4, 96, 96, 192),
#             r4 = ResidualBlock2(1, 192, 96, 96),
#             r5 = ResidualBlock2(1, 96, 48, 48),
#             r6 = ResidualBlock2(1, 48, 24, 24),
#             c7 = L.Convolution2D(24, 1, 3, stride=1, pad=1, initialW = initializers.HeNormal()),
#         )
#
#     def __call__(self, x):
#         h = self.c0(x)
#         #h = self.r0(h)
#         h = self.r1(h)
#         h = self.r2(h)
#         h = self.r3(h)
#         h = self.r4(h)
#         h = self.r5(h)
#         h = self.r6(h)
#         h = self.c7(h)
#         return F.tanh(h)#F.sigmoid(h)*255 #(1+F.tanh(h))/2 #
#         #return 2*F.sigmoid(x)-1
#         #return (1+F.tanh(h))*127.5

# class Generator(chainer.Chain):
#     def __init__(self):
#         super(Generator, self).__init__(
#             c0 = DilatedConvBlock(1, 32, ksize=3, stride=1, pad=1, dilated=2),
#             c1 = DilatedConvBlock(32, 64, ksize=3, stride=2, pad=1, dilated=2),
#             c2 = DilatedConvBlock(64, 128, ksize=3, stride=2, pad=1, dilated=2),
#             c3 = DilatedConvBlock(128, 256, ksize=3, stride=2, pad=1, dilated=2),
#             # r = ResidualBlock1(5, 256, 128, 256, stride=1),
#             c4 = DilatedConvBlock(256, 128, ksize=4, stride=2, pad=1, dilated=2),
#             c5 = DilatedConvBlock(128, 64, ksize=4, stride=2, pad=1, dilated=2),
#             c6 = DilatedConvBlock(64, 32, ksize=4, stride=2, pad=1, dilated=2),
#             c7 = L.Convolution2D(32, 1, 3, stride=1, pad=1, initialW = initializers.HeNormal()),
#         )
#
#     def __call__(self, x):
#         h = self.c3(self.c2(self.c1(self.c0(x))))
#         # h = F.dropout(self.r(h))
#         h = self.c7(self.c6(self.c5(self.c4(h))))
#         return F.tanh(h)

class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            c0 = ConvBlock(1, 32, ksize=3, stride=1, pad=1),
            c1 = ConvBlock(32, 64, ksize=3, stride=2, pad=1),
            c2 = ConvBlock(64, 128, ksize=3, stride=2, pad=1),
            c3 = ConvBlock(128, 256, ksize=3, stride=2, pad=1),
            r = ResidualBlock1(5, 256, 128, 256, stride=1),
            c4 = ConvBlock(256, 128, ksize=4, stride=2, pad=1, cfun=L.Deconvolution2D),
            c5 = ConvBlock(128, 64, ksize=4, stride=2, pad=1, cfun=L.Deconvolution2D),
            c6 = ConvBlock(64, 32, ksize=4, stride=2, pad=1, cfun=L.Deconvolution2D),
            c7 = L.Convolution2D(32, 1, 3, stride=1, pad=1, initialW = initializers.HeNormal()),
        )

    def __call__(self, x):
        h = self.c3(self.c2(self.c1(self.c0(x))))
        h = F.dropout(self.r(h))
        h = self.c7(self.c6(self.c5(self.c4(h))))
        return F.tanh(h) #F.sigmoid(h)*255 #(1+F.tanh(h))/2 #F.tanh(h)#
        #return 2*F.sigmoid(x)-1
        #return (1+F.tanh(h))*127.5


# class Generator(chainer.Chain):
#     def __init__(self):
#         super(Generator, self).__init__(
#             c0 = ConvBlock(1, 64, ksize=7, stride=1, pad=3),
#             r0 = ResidualBlock1(3, 64, 32, 16, stride=1),
#             r1 = ResidualBlock1(3, 64, 32, 16, stride=1),
#             r2 = ResidualBlock1(3, 64, 32, 16, stride=1),
#             r3 = ResidualBlock1(3, 64, 32, 16, stride=1),
#             r5 = ResidualBlock1(3, 64, 32, 16, stride=1),
#             r6 = ResidualBlock1(3, 64, 32, 16, stride=1),
#             r4 = ResidualBlock1(3, 96, 64, 32, stride=1),
#             # r4 = ResidualBlock1(3, 752, 64, 32, stride=1),
#             c7 = L.Deconvolution2D(32, 1, 7, stride=1, pad=3, initialW = initializers.HeNormal()),
#         )
#
#     def __call__(self, x):
#         h = self.c0(F.dropout(x, 0.05))
#         h0 = self.r0(h)
#         # h = self.c0(x)
#         h1 = self.r1(F.average_pooling_2d(h,(2,2)))
#         h2 = self.r2(F.average_pooling_2d(h,(4,4)))
#         h3 = self.r3(F.average_pooling_2d(h,(8,8)))
#         h4 = self.r2(F.average_pooling_2d(h,(16,16)))
#         h5 = self.r3(F.average_pooling_2d(h,(32,32)))
#         # h = F.dropout(self.r(F.dropout(h)))
#         # h = F.dropout(self.r(h))
#         # h = self.r(h)
#         h5 = F.resize_images(h5,h0.shape[2:])
#         h4 = F.resize_images(h4,h0.shape[2:])
#         h3 = F.resize_images(h3,h0.shape[2:])
#         h2 = F.resize_images(h2,h0.shape[2:])
#         h1 = F.resize_images(h1,h0.shape[2:])
#         h = F.concat([h0,h1,h2,h3,h4,h5],1)
#         # h = F.concat([h0,h1,h2,h3],1)
#         h = self.c7(self.r4(h))
#         return F.tanh(h)


class VGG(chainer.Chain):
    def __init__(self):
        super(VGG, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),
        )

    def preprocess(self, image):
        return image*127.5

    def __call__(self, x):
        x = self.preprocess(x)
        x = F.concat([x, x, x], axis=1)
        
        y11 = F.relu(self.conv1_1(x))
        y12 = F.relu(self.conv1_2(y11))
        h = F.max_pooling_2d(y12, 2, stride=2)
        
        y21 = F.relu(self.conv2_1(h))
        y22 = F.relu(self.conv2_2(y21))
        h = F.max_pooling_2d(y22, 2, stride=2)
        
        y31 = F.relu(self.conv3_1(h))
        y32 = F.relu(self.conv3_2(y31))
        y33 = F.relu(self.conv3_3(y32))
        h = F.max_pooling_2d(y33, 2, stride=2)
        
        y41 = F.relu(self.conv4_1(h))
        y42 = F.relu(self.conv4_2(y41))
        y43 = F.relu(self.conv4_3(y42))
        h = F.max_pooling_2d(y43, 2, stride=2)
        
        y51 = F.relu(self.conv5_1(h))
        y52 = F.relu(self.conv5_2(y51))
        y53 = F.relu(self.conv5_3(y52))
        h = F.max_pooling_2d(y53, 2, stride=2)
        # return  [y11,y12,y21,y22,y31,y32,y33,y41,y42,y43,y51,y52,y53]
        # return [y52, y42, y32, y22, y12]
        # return [y11,y21,y31,y41,y51]
        y1 = F.concat([y11, y12], axis=1)
        y2 = F.concat([y21, y22], axis=1)
        y3 = F.concat([y31, y32, y33], axis=1)
        y4 = F.concat([y41, y42, y43], axis=1)
        y5 = F.concat([y51, y52, y53], axis=1)
        
        return [y12,y22,y32,y42,y52]
        # return [y1, y2, y3, y4, y5]
        # return [y32,y42]
        # return y51
        # return y21
        # return h

class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__()
        with self.init_scope():
            '''
            self.conv0 = ConvBlock(64, 64, 1,1,0)
            self.conv1 = ConvBlock(128, 64, 1,1,0)
            self.conv2 = ConvBlock(256, 64, 1,1,0)
            self.conv3 = ConvBlock(512, 64, 1,1,0)
            self.conv4 = ConvBlock(512, 64, 1,1,0)
            '''
            self.conv0 = ConvBlock(128, 64, 1,1,0)
            self.conv1 = ConvBlock(256, 64, 1,1,0)
            self.conv2 = ConvBlock(768, 64, 1,1,0)
            self.conv3 = ConvBlock(1536, 64, 1,1,0)
            self.conv4 = ConvBlock(1536, 64, 1,1,0)
            """
            self.conv0 = ConvBlock(128, 64, 2,1,0)
            self.conv1 = ConvBlock(256, 64, 2,1,0)
            self.conv2 = ConvBlock(768, 64, 2,1,0)
            self.conv3 = ConvBlock(1536, 64, 2,1,0)
            self.conv4 = ConvBlock(1536, 64, 2,1,0)
            """
            """
            self.conv0 = ConvBlock(128, 64, 3, 1, 1)
            self.conv1 = ConvBlock(256, 64, 3, 1, 1)
            self.conv2 = ConvBlock(768, 64, 3, 1, 1)
            self.conv3 = ConvBlock(1536, 64, 3, 1, 1)
            self.conv4 = ConvBlock(1536, 64, 3, 1, 1)
            """
            # (n_in, n_out, ksize, stride, pad
            # self.convs = [self.conv0,self.conv1,self.conv2,self.conv3,self.conv4]
            self.lin = L.Linear(None, 1)

    def __call__(self, x):
        y0 = self.conv0(F.normalize(x[0]))      #fusion-attention
        y1 = self.conv1(F.normalize(x[1]))
        y2 = self.conv2(F.normalize(x[2]))
        y3 = self.conv3(F.normalize(x[3]))
        y4 = self.conv4(F.normalize(x[4]))
        
        pred = self.lin(F.concat([F.average(y0,(2,3)),F.average(y1,(2,3)),F.average(y2,(2,3)),F.average(y3,(2,3)),
                                  F.average(y4,(2,3))],1))
        return pred, [y0,y1,y2,y3,y4]


# class Discriminator(chainer.Chain):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         with self.init_scope():
#             # self.conv0 = ConvBlock(64, 32, rfun=F.leaky_relu)
#             self.conv1 = ConvBlock(128, 32, rfun=F.leaky_relu)
#             self.conv2 = ConvBlock(256, 32, rfun=F.leaky_relu)
#             self.conv3 = ConvBlock(512, 32, rfun=F.leaky_relu)
#             self.conv4 = ConvBlock(512, 32, rfun=F.leaky_relu)
#             self.convs = [self.conv1,self.conv2,self.conv3,self.conv4]
#             self.conv = ConvBlock(128, 64, rfun=F.leaky_relu)
#             self.lin = L.Linear(None,1)
#
#
#     def __call__(self, x1, x2):
#         diffs = []
#         feats0 = []
#         feats1 = []
#
#         for (kk,out) in enumerate(x1):
#             feats0.append(self.convs[kk](F.dropout(out)))
#             feats1.append(self.convs[kk](F.dropout(x2[kk])))
#             diffs.append(F.mean((feats0[kk]-feats1[kk])**2,(2,3)))
#         h = F.concat(diffs,1)
#         val = F.mean(h,1)
#         pred = self.lin(h)
#         return pred, val


# class Discriminator(chainer.Chain):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         with self.init_scope():
#             self.conv0 = ConvBlock(64, 32, rfun=F.leaky_relu)
#             self.conv1 = ConvBlock(128, 32, rfun=F.leaky_relu)
#             self.conv2 = ConvBlock(256, 32, rfun=F.leaky_relu)
#             self.conv3 = ConvBlock(512, 32, rfun=F.leaky_relu)
#             self.conv4 = ConvBlock(512, 32, rfun=F.leaky_relu)
#             self.convs = [self.conv0, self.conv1, self.conv2, self.conv3, self.conv4]
#             self.conv = ConvBlock(320, 64, rfun=F.leaky_relu)
#             self.lin = L.Linear(None, 1)
#
#     def __call__(self, x1, x2):
#         diffs = []
#         feats0 = []
#         feats1 = []
#
#         # for (kk,out) in enumerate(x1):
#         #     feats0.append(self.convs[kk](F.dropout(out)))
#         #     feats1.append(self.convs[kk](F.dropout(x2[kk])))
#         #     diffs.append(F.mean((feats0[kk]-feats1[kk])**2,(2,3)))
#         # h = F.concat(diffs,1)
#         # val = F.mean(h,1)
#         # h = self.lin(h)
#
#         for (kk, out) in enumerate(x1):
#             feats0.append(F.resize_images(self.convs[kk](F.dropout(out)), (x1[0].shape[0],x1[0].shape[1])))
#             feats1.append(F.resize_images(self.convs[kk](F.dropout(x2[kk])), (x1[0].shape[0],x1[0].shape[1])))
#         h0 = F.concat(feats0, 1)
#         h1 = F.concat(feats1, 1)
#         diff = F.mean((h0 - h1) ** 2, (2, 3))
#         val = F.mean(diff, 1)
#         # h = self.conv(F.concat((h0,h1),1))
#         # h = self.conv(h0 - h1)
#         pred = self.lin(diff)
#         return pred, val

# class Discriminator(chainer.Chain):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         with self.init_scope():
#             self.conv0 = ConvBlock(128, 32, rfun=F.leaky_relu)
#             self.conv1 = ConvBlock(256, 32, rfun=F.leaky_relu)
#             self.conv2 = ConvBlock(512, 32, rfun=F.leaky_relu)
#             self.conv3 = ConvBlock(1024, 32, rfun=F.leaky_relu)
#             self.conv4 = ConvBlock(1024, 32, rfun=F.leaky_relu)
#             self.convs = [self.conv0,self.conv1,self.conv2,self.conv3,self.conv4]
#             self.conv = ConvBlock(160, 64, rfun=F.leaky_relu)
#             self.lin = L.Linear(None,1)
#
#
#     def __call__(self, x1, x2):
#         diffs = []
#         feats=[]
#
#         for (kk,out) in enumerate(x1):
#             h = self.convs[kk](F.dropout(F.concat((out,x2[kk]),1)))
#             diffs.append(h)
#             feats.append(F.resize_images(h,x1[-1].shape))
#         h = self.conv(F.concat(feats,1))
#         val = F.mean(feats[0],(2,3))+F.mean(feats[1],(2,3))+F.mean(feats[2],(2,3))+F.mean(feats[3],(2,3))+F.mean(feats[4],(2,3))#F.mean(h,1)
#         pred = self.lin(h)
#         return pred, val

# class Discriminator(chainer.Chain):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         with self.init_scope():
#             self.conv0 = ConvBlock(64, 32)#, rfun=F.leaky_relu)
#             self.conv1 = ConvBlock(128, 32)#, rfun=F.leaky_relu)
#             self.conv2 = ConvBlock(256, 32)#, rfun=F.leaky_relu)
#             self.conv3 = ConvBlock(512, 32)#, rfun=F.leaky_relu)
#             self.conv4 = ConvBlock(512, 32)#, rfun=F.leaky_relu)
#             self.convs = [self.conv0,self.conv1,self.conv2,self.conv3,self.conv4]
#             self.conv = ConvBlock(160, 64, rfun=F.leaky_relu)
#             self.lin = L.Linear(None,1)
#
#     def __call__(self, x1, x2):
#         diffs = []
#         feats=[]
#
#         for (kk,out) in enumerate(x1):
#             out1 = F.normalize(out)
#             out2 = F.normalize(x2[kk])
#             diffs.append(F.mean((out1-out2)**2,(1,2,3)))
#             h1 = self.convs[kk](F.dropout(out1))
#             h2 = self.convs[kk](F.dropout(out2))
#             feats.append(F.resize_images(h1-h2,x1[-1].shape))
#         # h = self.conv(F.concat(feats,1))
#         val = diffs[0]+diffs[1]+diffs[2]+diffs[3]+diffs[4]#F.mean(h,1)
#         pred = self.lin(F.concat(feats,1))
#         return pred, val

# class Discriminator(chainer.Chain):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         with self.init_scope():
#             self.conv0 = ConvBlock(64, 32)#, rfun=F.leaky_relu)
#             self.conv1 = ConvBlock(128, 32)#, rfun=F.leaky_relu)
#             self.conv2 = ConvBlock(256, 32)#, rfun=F.leaky_relu)
#             self.conv3 = ConvBlock(512, 32)#, rfun=F.leaky_relu)
#             self.conv4 = ConvBlock(512, 32)#, rfun=F.leaky_relu)
#             self.convs = [self.conv0,self.conv1,self.conv2,self.conv3,self.conv4]
#             self.conv = ConvBlock(160, 64, rfun=F.leaky_relu)
#             self.lin = L.Linear(None,1)
#
#     def __call__(self, x1, x2):
#         diffs = []
#         feats=[]
#
#         for (kk,out) in enumerate(x1):
#             out1 = F.normalize(out)
#             out2 = F.normalize(x2[kk])
#             h1 = self.convs[kk](F.dropout((out1-out2)**2))
#             feats.append(F.resize_images(h1,x1[-1].shape))
#         # h = self.conv(F.concat(feats,1))
#         val = 0
#         pred = self.lin(F.concat(feats,1))
#         return pred, val

# class Discriminator(chainer.Chain):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         with self.init_scope():
#             # self.conv0 = ConvBlock(64, 32)#, rfun=F.leaky_relu)
#             self.conv1 = ConvBlock(128, 64)#, rfun=F.leaky_relu)
#             self.conv2 = ConvBlock(256, 128)#, rfun=F.leaky_relu)
#             self.conv3 = ConvBlock(512, 256)#, rfun=F.leaky_relu)
#             self.conv4 = ConvBlock(512, 512)#, rfun=F.leaky_relu)
#             self.convs = [self.conv4,self.conv3,self.conv2,self.conv1]
#             self.conv = ConvBlock(64, 16)#, rfun=F.leaky_relu)
#             self.lin = L.Linear(None,1)
#
#     def __call__(self, x1,x2):
#         # feats = []
#         h = x1[0]-x2[0]
#         for(kk,out) in enumerate(x1):
#             if kk < len(x1)-1:
#                 h = F.resize_images(self.convs[kk](h),x1[kk+1].shape[2:])#x1[kk+1]
#                 h = x1[kk+1]-x2[kk+1]+h
#                 #feats.append(h)
#         pred = self.lin(self.conv(h))
#         return pred


# class Discriminator(chainer.Chain):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         with self.init_scope():
#             self.conv0 = ConvBlock(64, 32)#, rfun=F.leaky_relu)
#             self.conv1 = ConvBlock(128, 32)#, rfun=F.leaky_relu)
#             self.conv2 = ConvBlock(256, 32)#, rfun=F.leaky_relu)
#             self.conv3 = ConvBlock(512, 32)#, rfun=F.leaky_relu)
#             self.conv4 = ConvBlock(512, 32)#, rfun=F.leaky_relu)
#             self.convs = [self.conv0,self.conv1,self.conv2,self.conv3,self.conv4]
#             self.conv = ConvBlock(160, 16)#, rfun=F.leaky_relu)
#             self.lin = L.Linear(None,1)
#
#     def __call__(self, x1,x2):
#         feats = []
#         for(kk,out) in enumerate(x1):
#             # out1 = self.convs[kk](F.normalize(out))
#             # out2 = self.convs[kk](F.normalize(x2[kk]))
#             # h = F.mean(out1-out2, (2,3))
#         #     out1 = self.convs[kk](F.normalize(out)-F.normalize(x2[kk]))
#         #     h = F.mean(out1, (2,3))
#         #     feats.append(h)
#         # # pred = self.lin(self.conv(h))
#         # pred = self.lin(F.concat(feats,1))
#             h = F.resize_images(self.convs[kk](F.normalize(out)-F.normalize(x2[kk])),x1[-1].shape[2:])
#             # h = F.mean(h, (2,3))
#             feats.append(h)
#         pred = self.lin(self.conv(F.concat(feats,1)))
#         # pred = self.lin(F.concat(feats,1))
#         return pred

# class Discriminator(chainer.Chain):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         with self.init_scope():
#             # self.conv0 = ConvBlock(64, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv1 = ConvBlock(128, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv2 = ConvBlock(256, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv3 = ConvBlock(512, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv4 = ConvBlock(512, 32, 3,1,1)#, rfun=F.leaky_relu)
#             self.conv0 = ConvBlock(64, 32, 1,1,0)#, rfun=F.leaky_relu)
#             self.conv1 = ConvBlock(128, 32, 1,1,0)#, rfun=F.leaky_relu)
#             self.conv2 = ConvBlock(256, 32, 1,1,0)#, rfun=F.leaky_relu)
#             self.conv3 = ConvBlock(512, 32, 1,1,0)#, rfun=F.leaky_relu)
#             self.conv4 = ConvBlock(512, 32, 1,1,0)#, rfun=F.leaky_relu)
#             self.convs = [self.conv0,self.conv1,self.conv2,self.conv3,self.conv4]
#             # self.convs = [self.conv2, self.conv3]
#             self.conv = ConvBlock(160, 5, 3,1,1)#, rfun=F.leaky_relu)
#             # # self.convl = ConvBlock(128, 64, 6, 2, 1)
#             # # self.lin1 = L.Linear(None,512)
#             # self.conv = ConvBlock(512, 128,1,1,0)#, rfun=F.leaky_relu)
#             # self.conv = ConvBlock(5, 1, 1, 1, 0)
#             self.lin0 = L.Linear(None, 1)
#             self.lin1 = L.Linear(None, 1)
#             self.lin2 = L.Linear(None, 1)
#             self.lin3 = L.Linear(None, 1)
#             self.lin4 = L.Linear(None, 1)
#             # self.lin = L.Linear(None, 1)
#
#     # def __call__(self, x1, x2):
#     #     feats = []
#     #     pred = self.lin(self.conv(x1-x2))
#     #     return pred, feats
#
#     def __call__(self, x):
#         y0 = self.convs[0](F.resize_images(F.normalize(x[0]),x[2].shape[2:]))      #fusion-attention
#         y1 = self.convs[1](F.resize_images(F.normalize(x[1]),y0.shape[2:]))
#         y2 = self.convs[2](F.resize_images(F.normalize(x[2]),y0.shape[2:]))
#         y3 = self.convs[3](F.resize_images(F.normalize(x[3]),y0.shape[2:]))
#         y4 = self.convs[4](F.resize_images(F.normalize(x[4]),y0.shape[2:]))
#         # val = hs[0] + hs[1] + hs[2] + hs[3] + hs[4]
#         # hs = F.expand_dims(F.expand_dims(F.concat(hs,1),2),3)
#         # pred = self.conv(hs)
#         atten = F.softmax(self.conv(F.concat((y0,y1,y2,y3,y4),1)),1)
#         at_split = F.split_axis(atten,5,axis=1)
#         # at_split = F.split_axis(atten,(32,64,96,128),axis=1)
#         y0 = F.normalize(x[0])*F.resize_images(F.repeat(at_split[1],x[0].shape[1],axis=1),x[0].shape[2:])
#         y1 = F.normalize(x[1])*F.resize_images(F.repeat(at_split[1],x[1].shape[1],axis=1),x[1].shape[2:])
#         y2 = F.normalize(x[2])*F.resize_images(F.repeat(at_split[2],x[2].shape[1],axis=1),x[2].shape[2:])
#         y3 = F.normalize(x[3])*F.resize_images(F.repeat(at_split[3],x[3].shape[1],axis=1),x[3].shape[2:])
#         y4 = F.normalize(x[4])*F.resize_images(F.repeat(at_split[4],x[4].shape[1],axis=1),x[4].shape[2:])
#         pred = F.concat([self.lin0(y0),self.lin1(y1),self.lin2(y2),self.lin3(y3),self.lin4(y4)], 1)
#         all = [y0,y1,y2,y3,y4]
#
#
#         # y0 = F.max_pooling_2d(F.normalize(x[0]),4)     #fusion-attention
#         # y1 = F.max_pooling_2d(F.normalize(x[1]),2)
#         # y2 = F.normalize(x[2])
#         # y3 = F.resize_images(F.normalize(x[3]),y2.shape[2:])
#         # y4 = F.resize_images(F.normalize(x[4]),y2.shape[2:])
#         # # val = hs[0] + hs[1] + hs[2] + hs[3] + hs[4]
#         # # hs = F.expand_dims(F.expand_dims(F.concat(hs,1),2),3)
#         # # pred = self.conv(hs)
#         # all = F.concat((self.conv0(y0),self.conv1(y1),self.conv2(y2),self.conv3(y3),self.conv4(y4)),1)
#         # atten = F.softmax(self.conv(all),1)
#         # # all = all*F.resize_images(atten,all.shape[2:])
#         # all = F.concat((y0,y1,y2,y3,y4),1)*F.resize_images(atten,y2.shape[2:])
#         # pred = self.lin(all)
#
#         # hs = []
#
#         # # y0 = F.max_pooling_2d(self.convs[0](F.normalize(x[0]),4))      #fusion-attention
#         # # y1 = F.max_pooling_2d(self.convs[1](F.normalize(x[1]),2))
#         # # y2 = self.convs[2](F.normalize(x[2]))
#         # # y3 = F.resize_images(self.convs[3](F.normalize(x[3]),y2.shape[2:]))
#         # # y4 = F.resize_images(self.convs[4](F.normalize(x[4]),y2.shape[2:]))
#         # # y0 = self.convs[0](F.normalize(x[0]))      #fusion-attention
#         # # y1 = self.convs[1](F.normalize(x[1]))
#         # # y2 = self.convs[2](F.normalize(x[2]))
#         # # y3 = self.convs[3](F.normalize(x[3]))
#         # # y4 = self.convs[4](F.normalize(x[4]))
#         # y0 = self.conv0(F.normalize(x[0]))      #fusion-attention
#         # y1 = self.conv1(F.normalize(x[1]))
#         # y2 = self.conv2(F.normalize(x[2]))
#         # y3 = self.conv3(F.normalize(x[3]))
#         # y4 = self.conv4(F.normalize(x[4]))
#         # # all = F.concat((F.max_pooling_2d(y0,16),F.max_pooling_2d(y1,8),F.max_pooling_2d(y2,4),F.max_pooling_2d(y3,2),y4),1)
#         # # all = F.concat((F.max_pooling_2d(y0,8),F.max_pooling_2d(y1,4),F.max_pooling_2d(y2,2),y3,F.resize_images(y4,y3.shape[2:])),1)
#         # all = F.concat((F.max_pooling_2d(y0,4),F.max_pooling_2d(y1,2),y2,F.resize_images(y3,y2.shape[2:]),F.resize_images(y4,y2.shape[2:])),1)
#         # # val = hs[0] + hs[1] + hs[2] + hs[3] + hs[4]
#         # # hs = F.expand_dims(F.expand_dims(F.concat(hs,1),2),3)
#         # # pred = self.conv(hs)
#         # atten = F.softmax(F.average(self.conv(all),(2,3),keepdims=True),1)
#         # # atten = F.softmax(self.lin2(F.average(all,(2,3))),1)
#         # # atten = F.softmax(self.conv(all),1)
#         # all = all*F.resize_images(atten,all.shape[2:])
#         # # at_split = F.split_axis(atten,5,axis=1)
#         # # at_split = F.split_axis(atten,(32,64,96,128,160),axis=1)
#         # # y0 = y0*F.resize_images(at_split[0],x[0].shape[2:])
#         # # y1 = y1*F.resize_images(at_split[1],x[1].shape[2:])
#         # # y2 = y2*F.resize_images(at_split[2],x[2].shape[2:])
#         # # y3 = y3*F.resize_images(at_split[3],x[3].shape[2:])
#         # # y4 = y4*F.resize_images(at_split[4],x[4].shape[2:])
#         # # y0 = y0*F.resize_images(F.repeat(at_split[0],y0.shape[1],axis=1),x[0].shape[2:])
#         # # y1 = y1*F.resize_images(F.repeat(at_split[1],y1.shape[1],axis=1),x[1].shape[2:])
#         # # y2 = y2*F.resize_images(F.repeat(at_split[2],y2.shape[1],axis=1),x[2].shape[2:])
#         # # y3 = y3*F.resize_images(F.repeat(at_split[3],y3.shape[1],axis=1),x[3].shape[2:])
#         # # y4 = y4*F.resize_images(F.repeat(at_split[4],y4.shape[1],axis=1),x[4].shape[2:])
#         # # hs = [h0,h1,h2,h3,h4]
#         # # pred = F.concat([self.lin0(y0),self.lin1(y1),self.lin2(y2),self.lin3(y3),self.lin4(y4)],1)
#         # # pred = self.lin(F.concat([F.average(y0,(2,3)),F.average(y1,(2,3)),F.average(y2,(2,3)),F.average(y3,(2,3)),F.average(y4,(2,3))],1))
#         # # pred = self.lin(F.average(y0,(2,3))+F.average(y1,(2,3))+F.average(y2,(2,3))+F.average(y3,(2,3))+F.average(y4,(2,3)))
#         # pred = self.lin(all)
#         # val = y0+y1+y2+y3+y4
#         # pred = F.average(val,1)
#         # val = hs[0]+hs[1]+hs[2]+hs[3]+hs[4]
#         # pred = self.lin(val)
#         return pred, all#[y0,y1,y2,y3,y4]#, atten


# class Discriminator(chainer.Chain):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         with self.init_scope():
#             # self.conv0 = ConvBlock(64, 64, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv1 = ConvBlock(128, 128, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv2 = ConvBlock(256, 256, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv3 = ConvBlock(512, 512, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv4 = ConvBlock(512, 512, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv0 = ConvBlock(64, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv1 = ConvBlock(128, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv2 = ConvBlock(256, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv3 = ConvBlock(512, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv4 = ConvBlock(512, 32, 3,1,1)#, rfun=F.leaky_relu)
#             self.conv0 = ConvBlock(64, 64, 1,1,0)#, rfun=F.leaky_relu)
#             self.conv1 = ConvBlock(128, 64, 1,1,0)#, rfun=F.leaky_relu)
#             self.conv2 = ConvBlock(256, 64, 1,1,0)#, rfun=F.leaky_relu)
#             self.conv3 = ConvBlock(512, 64, 1,1,0)#, rfun=F.leaky_relu)
#             self.conv4 = ConvBlock(512, 64, 1,1,0)#, rfun=F.leaky_relu)
#             self.convs = [self.conv0,self.conv1,self.conv2,self.conv3,self.conv4]
#             # # self.convs = [self.conv2, self.conv3]
#             # self.conv = ConvBlock(160, 5, 3,1,1)#, rfun=F.leaky_relu)
#             # # self.convl = ConvBlock(128, 64, 6, 2, 1)
#             # # self.lin1 = L.Linear(None,512)
#             # self.conv = ConvBlock(512, 128,1,1,0)#, rfun=F.leaky_relu)
#             # self.conv = ConvBlock(5, 1, 1, 1, 0)
#             # self.lin0 = L.Linear(None, 1)
#             # self.lin1 = L.Linear(None, 1)
#             # self.lin2 = L.Linear(None, 1)
#             # self.lin3 = L.Linear(None, 1)
#             # self.lin4 = L.Linear(None, 1)
#             self.lin = L.Linear(None, 1)
#
#     # def __call__(self, x1, x2):
#     #     feats = []
#     #     pred = self.lin(self.conv(x1-x2))
#     #     return pred, feats
#
#     def __call__(self, x):
#         y0 = self.conv0(F.normalize(x[0]))      #fusion-attention
#         y1 = self.conv1(F.normalize(x[1]))
#         y2 = self.conv2(F.normalize(x[2]))
#         y3 = self.conv3(F.normalize(x[3]))
#         y4 = self.conv4(F.normalize(x[4]))
#         pred = self.lin(F.concat([F.average(y0,(2,3)),F.average(y1,(2,3)),F.average(y2,(2,3)),F.average(y3,(2,3)),F.average(y4,(2,3))],1))
#         # val = F.average(y0,(2,3))+F.average(y1,(2,3))+F.average(y2,(2,3))+F.average(y3,(2,3))+F.average(y4,(2,3))
#         # val = y0+y1+y2+y3+y4
#         # pred = F.average(val,1)
#         # val = hs[0]+hs[1]+hs[2]+hs[3]+hs[4]
#         # pred = self.lin(val)
#         return pred, [y0,y1,y2,y3,y4]#, atten


# class Discriminator(chainer.Chain):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         with self.init_scope():
#             self.conv0 = ConvBlock(64, 64, 1, 1, 0)  # , rfun=F.leaky_relu)
#             self.conv1 = ConvBlock(64, 64, 1, 1, 0)  # , rfun=F.leaky_relu)
#             self.conv2 = ConvBlock(128, 64, 1, 1, 0)  # , rfun=F.leaky_relu)
#             self.conv3 = ConvBlock(128, 64, 1, 1, 0)  # , rfun=F.leaky_relu)
#             self.conv4 = ConvBlock(256, 64, 1, 1, 0)  # , rfun=F.leaky_relu)
#             self.conv5 = ConvBlock(256, 64, 1, 1, 0)  # , rfun=F.leaky_relu)
#             self.conv6 = ConvBlock(256, 64, 1, 1, 0)  # , rfun=F.leaky_relu)
#             self.conv7 = ConvBlock(512, 64, 1, 1, 0)  # , rfun=F.leaky_relu)
#             self.conv8 = ConvBlock(512, 64, 1, 1, 0)  # , rfun=F.leaky_relu)
#             self.conv9 = ConvBlock(512, 64, 1, 1, 0)  # , rfun=F.leaky_relu)
#             self.conv10 = ConvBlock(512, 64, 1, 1, 0)  # , rfun=F.leaky_relu)
#             self.conv11 = ConvBlock(512, 64, 1, 1, 0)  # , rfun=F.leaky_relu)
#             self.conv12 = ConvBlock(512, 64, 1, 1, 0)  # , rfun=F.leaky_relu)
#             # self.convs = [self.conv0,self.conv1,self.conv2,self.conv3,self.conv4,self.conv5,self.conv6,self.conv7,self.conv8,self.conv9,self.conv10,self.conv11,self.conv12]
#             self.conv = ConvBlock(832, 832, 1, 1, 0)
#             # self.conv = ConvBlock(832, 832, 3, 1, 1)
#             self.lin = L.Linear(None, 1)
#             # self.conv0 = ConvBlock(64, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv1 = ConvBlock(64, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv2 = ConvBlock(128, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv3 = ConvBlock(128, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv4 = ConvBlock(256, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv5 = ConvBlock(256, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv6 = ConvBlock(256, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv7 = ConvBlock(512, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv8 = ConvBlock(512, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv9 = ConvBlock(512, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv10 = ConvBlock(512, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv11 = ConvBlock(512, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv12 = ConvBlock(512, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # # self.convs = [self.conv0,self.conv1,self.conv2,self.conv3,self.conv4,self.conv5,self.conv6,self.conv7,self.conv8,self.conv9,self.conv10,self.conv11,self.conv12]
#             # self.conv = ConvBlock(416,416,1,1,0)
#             # self.lin = L.Linear(None, 1)
#
#
#     # def __call__(self, x1, x2):
#     #     feats = []
#     #     pred = self.lin(self.conv(x1-x2))
#     #     return pred, feats
#
#     def __call__(self, x):
#         # # hs = []
#         # y11 = F.resize_images(self.conv0(F.normalize(x[0])),x[4].shape[2:])     #fusion-attention
#         # y12 = F.resize_images(self.conv1(F.normalize(x[1])),x[4].shape[2:])     #fusion-attention
#         # y21 = F.resize_images(self.conv2(F.normalize(x[2])),x[4].shape[2:])
#         # y22 = F.resize_images(self.conv3(F.normalize(x[3])),x[4].shape[2:])
#         # y31 = self.conv4(F.normalize(x[4]))
#         # y32 = self.conv5(F.normalize(x[5]))
#         # y33 = self.conv6(F.normalize(x[6]))
#         # y41 = F.resize_images(self.conv7(F.normalize(x[7])),x[4].shape[2:])
#         # y42 = F.resize_images(self.conv8(F.normalize(x[8])),x[4].shape[2:])
#         # y43 = F.resize_images(self.conv9(F.normalize(x[9])),x[4].shape[2:])
#         # y51 = F.resize_images(self.conv10(F.normalize(x[10])),x[4].shape[2:])
#         # y52 = F.resize_images(self.conv11(F.normalize(x[11])),x[4].shape[2:])
#         # y53 = F.resize_images(self.conv12(F.normalize(x[12])),x[4].shape[2:])
#         # all = self.conv(F.concat((y11,y12,y21,y22,y31,y32,y33,y41,y42,y43,y51,y52,y53),1))
#         # # pred = self.lin(F.average(all,(2,3)))
#         # pred = self.lin(all)
#         # return pred, all#[y11,y12,y21,y22,y31,y32,y33,y41,y42,y43,y51,y52,y53]#all
#
#         y11 = self.conv0(F.normalize(x[0]))     #fusion-attention
#         y12 = self.conv1(F.normalize(x[1]))     #fusion-attention
#         y21 = self.conv2(F.normalize(x[2]))
#         y22 = self.conv3(F.normalize(x[3]))
#         y31 = self.conv4(F.normalize(x[4]))
#         y32 = self.conv5(F.normalize(x[5]))
#         y33 = self.conv6(F.normalize(x[6]))
#         y41 = self.conv7(F.normalize(x[7]))
#         y42 = self.conv8(F.normalize(x[8]))
#         y43 = self.conv9(F.normalize(x[9]))
#         y51 = self.conv10(F.normalize(x[10]))
#         y52 = self.conv11(F.normalize(x[11]))
#         y53 = self.conv12(F.normalize(x[12]))
#         all = F.concat((F.average(y11, (2, 3)), F.average(y12, (2, 3)), F.average(y21, (2, 3)), F.average(y22, (2, 3)),
#                         F.average(y31, (2, 3)), F.average(y32, (2, 3)),
#                         F.average(y33, (2, 3)), F.average(y41, (2, 3)), F.average(y42, (2, 3)), F.average(y43, (2, 3)),
#                         F.average(y51, (2, 3)), F.average(y52, (2, 3)),
#                         F.average(y53, (2, 3))))
#         pred = self.lin(all)
#         return pred, x[10]#[y11, y12, y21, y22, y31, y32, y33, y41, y42, y43, y51, y52, y53]  # all

# class Discriminator(chainer.Chain):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         with self.init_scope():
#             self.conv0 = ConvBlock(64, 32, 3,1,1)#, rfun=F.leaky_relu)
#             self.conv1 = ConvBlock(128, 32, 3,1,1)#, rfun=F.leaky_relu)
#             self.conv2 = ConvBlock(256, 32, 3,1,1)#, rfun=F.leaky_relu)
#             self.conv3 = ConvBlock(512, 32, 3,1,1)#, rfun=F.leaky_relu)
#             self.conv4 = ConvBlock(512, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv0 = ConvBlock(64, 32, 1,1,0)#, rfun=F.leaky_relu)
#             # self.conv1 = ConvBlock(128, 32, 1,1,0)#, rfun=F.leaky_relu)
#             # self.conv2 = ConvBlock(256, 32, 1,1,0)#, rfun=F.leaky_relu)
#             # self.conv3 = ConvBlock(512, 32, 1,1,0)#, rfun=F.leaky_relu)
#             # self.conv4 = ConvBlock(512, 32, 1,1,0)#, rfun=F.leaky_relu)
#             self.convs = [self.conv0,self.conv1,self.conv2,self.conv3,self.conv4]
#             # # self.convs = [self.conv2, self.conv3]
#             self.conv = ConvBlock(160, 32,3,1,1)#, rfun=F.leaky_relu)
#             # # self.convl = ConvBlock(128, 64, 6, 2, 1)
#             # # self.lin1 = L.Linear(None,512)
#             # self.conv = ConvBlock(512, 128,1,1,0)#, rfun=F.leaky_relu)
#             # self.conv = ConvBlock(5, 1, 1, 1, 0)
#             self.lin = L.Linear(None,1)#, nobias=True)
#
#     # def __call__(self, x1, x2):
#     #     feats = []
#     #     pred = self.lin(self.conv(x1-x2))
#     #     return pred, feats
#
#     def __call__(self, x):
#         # hs = []
#         y0 = self.convs[0](F.average_pooling_2d(F.normalize(x[0]),4))      #weight2-concat
#         y1 = self.convs[1](F.average_pooling_2d(F.normalize(x[1]),2))
#         y2 = self.convs[2](F.normalize(x[2]))
#         y3 = self.convs[3](F.resize_images(F.normalize(x[3]),y2.shape[2:]))
#         y4 = self.convs[4](F.resize_images(F.normalize(x[4]),y2.shape[2:]))
#         # val = hs[0] + hs[1] + hs[2] + hs[3] + hs[4]
#         # hs = F.expand_dims(F.expand_dims(F.concat(hs,1),2),3)
#         # pred = self.conv(hs)
#         val = self.conv(F.concat((y0,y1,y2,y3,y4),1))
#         # val = y0+y1+y2+y3+y4
#         # pred = F.average(val,1)
#         # val = hs[0]+hs[1]+hs[2]+hs[3]+hs[4]
#         pred = self.lin(val)
#         return pred, val


# class Discriminator(chainer.Chain):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         with self.init_scope():
#             # self.dc5 = ConvBlock(512, 512, ksize=4, stride=2, pad=1, cfun=L.Deconvolution2D)
#             # self.dc4 = ConvBlock(512, 256, ksize=4, stride=2, pad=1, cfun=L.Deconvolution2D)
#             # self.dc3 = ConvBlock(256, 128, ksize=4, stride=2, pad=1, cfun=L.Deconvolution2D)
#             # self.dc2 = ConvBlock(128, 64, ksize=4, stride=2, pad=1, cfun=L.Deconvolution2D)
#             # self.dc0 = L.Deconvolution2D(64, 1, 3, stride=1, pad=1, initialW=initializers.HeNormal())
#             # self.conv0 = ConvBlock(64, 64, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv1 = ConvBlock(128, 128, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv2 = ConvBlock(256, 256, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv3 = ConvBlock(512, 512, 3,1,1)#, rfun=F.leaky_relu)
#             # self.conv4 = ConvBlock(512, 512, 3,1,1)#, rfun=F.leaky_relu)
#             self.conv0 = ConvBlock(64, 64, 1,1,0)#, rfun=F.leaky_relu)
#             self.conv1 = ConvBlock(128, 128, 1,1,0)#, rfun=F.leaky_relu)
#             self.conv2 = ConvBlock(256, 256, 1,1,0)#, rfun=F.leaky_relu)
#             self.conv3 = ConvBlock(512, 512, 1,1,0)#, rfun=F.leaky_relu)
#             self.conv4 = ConvBlock(512, 512, 1,1,0)#, rfun=F.leaky_relu)
#             # self.convs = [self.conv0,self.conv1,self.conv2,self.conv3,self.conv4]
#             # # self.convs = [self.conv2, self.conv3]
#             # self.conv = ConvBlock(160, 32,3,1,1)#, rfun=F.leaky_relu)
#             # # self.convl = ConvBlock(128, 64, 6, 2, 1)
#             # # self.lin1 = L.Linear(None,512)
#             # self.conv = ConvBlock(512, 128,1,1,0)#, rfun=F.leaky_relu)
#             # self.conv = ConvBlock(5, 1, 1, 1, 0)
#             self.lin = L.Linear(None,1)#, nobias=True)
#
#     # def __call__(self, x1, x2):
#     #     feats = []
#     #     pred = self.lin(self.conv(x1-x2))
#     #     return pred, feats
#
#     def __call__(self, x):
#         # hs = []
#         # dec = self.dc5(x[-1])
#         # dec = self.dc4(dec)
#         # dec = self.dc3(dec)
#         # dec = self.dc2(dec)
#         # dec = self.dc0(dec)
#         # y0 = self.convs[0](F.normalize(x[0]))      #weight2-concat
#         # y1 = self.convs[1](F.normalize(x[1]))
#         # y2 = self.convs[2](F.normalize(x[2]))
#         # y3 = self.convs[3](F.normalize(x[3]))
#         # y4 = self.convs[4](F.normalize(x[4]))
#         y0 = self.conv0(F.normalize(x[0]))      #weight2-concat
#         y1 = self.conv1(F.normalize(x[1]))
#         y2 = self.conv2(F.normalize(x[2]))
#         y3 = self.conv3(F.normalize(x[3]))
#         y4 = self.conv4(F.normalize(x[4]))
#         # val = hs[0] + hs[1] + hs[2] + hs[3] + hs[4]
#         # hs = F.expand_dims(F.expand_dims(F.concat(hs,1),2),3)
#         # pred = self.conv(hs)
#         # val = self.conv(F.concat((y0,y1,y2,y3,y4),1))
#         # val = y0+y1+y2+y3+y4
#         # val = F.average(y0,(2,3))+F.average(y1,(2,3))+F.average(y2,(2,3))+F.average(y3,(2,3))+F.average(y4,(2,3))
#         val = F.concat((F.average(y0,(2,3)),F.average(y1,(2,3)),F.average(y2,(2,3)),F.average(y3,(2,3)),F.average(y4,(2,3))),1)
#         # pred = F.average(val,1)
#         # val = hs[0]+hs[1]+hs[2]+hs[3]+hs[4]
#         pred = self.lin(val)
#         return pred, [y0,y1,y2,y3,y4]#,dec

# class Discriminator1(chainer.Chain):
#     def __init__(self):
#         super(Discriminator1, self).__init__()
#         with self.init_scope():
#             # self.conv0 = ConvBlock(64, 32,1,1,0)#, rfun=F.leaky_relu)
#             # self.conv1 = ConvBlock(128, 32,1,1,0)#, rfun=F.leaky_relu)
#             # self.conv2 = ConvBlock(256, 32,1,1,0)#, rfun=F.leaky_relu)
#             # self.conv3 = ConvBlock(512, 32,1,1,0)#, rfun=F.leaky_relu)
#             # self.conv4 = ConvBlock(512, 32,1,1,0)#, rfun=F.leaky_relu)
#             self.conva0 = ConvBlock(64, 32, 3,1,1)#, rfun=F.leaky_relu)
#             self.conva1 = ConvBlock(128, 32, 3,1,1)#, rfun=F.leaky_relu)
#             self.conva2 = ConvBlock(256, 32, 3,1,1)#, rfun=F.leaky_relu)
#             self.conva3 = ConvBlock(512, 32, 3,1,1)#, rfun=F.leaky_relu)
#             self.conva4 = ConvBlock(512, 32, 3,1,1)#, rfun=F.leaky_relu)
#             # self.convs = [self.conv0,self.conv1,self.conv2,self.conv3,self.conv4]
#             self.convas = [self.conva0,self.conva1,self.conva2,self.conva3,self.conva4]
#             # # self.convs = [self.conv2, self.conv3]
#             # self.conv = ConvBlock(320, 128,1,1,0)#, rfun=F.leaky_relu)
#             # # self.convl = ConvBlock(128, 64, 6, 2, 1)
#             # # self.lin1 = L.Linear(None,512)
#             # self.conv = ConvBlock(512, 128,1,1,0)#, rfun=F.leaky_relu)
#             self.lin = L.Linear(None,1)
#
#     # def __call__(self, x1, x2):
#     #     feats = []
#     #     pred = self.lin(self.conv(x1-x2))
#     #     return pred, feats
#
#     def __call__(self, x1,x2):
#         feats = []
#         hs=[]
#         hs2=[]
#         for(kk,out) in enumerate(x1):
#             # out1 = self.convs[kk](F.normalize(out))
#             # out2 = self.convs[kk](F.normalize(x2[kk]))
#             # h = F.mean(out1-out2, (2,3))
#         #     out1 = self.convs[kk](F.normalize(out)-F.normalize(x2[kk]))#mix
#         #     h = F.mean(out1, (2,3))
#         #     feats.append(h)
#         # # pred = self.lin(self.conv(h))
#         # diff = F.concat(feats,1)
#         # pred = self.lin(diff)
#         #     h = F.resize_images(self.convs[kk](F.normalize(out)-F.normalize(x2[kk])),x1[-1].shape[2:])#mix1
#         #     feats.append(h)
#         # pred = self.lin(F.concat(feats,1))
#         #     h = F.resize_images(self.convs[kk](F.normalize(out) - F.normalize(x2[kk])), x1[-1].shape)  # mix2 #vgg+->mix3
#         #     feats.append(h)
#         # pred = self.lin(self.conv(F.concat(feats, 1)))
#         # pred = self.lin(F.concat(feats, 1))
#         #     h = F.resize_images(F.normalize(out) - F.normalize(x2[kk]), x1[-1].shape)  # mix2 #vgg+->mix3
#         #     feats.append(h)
#         # pred = self.lin(self.conv(F.concat(feats, 1)))
#         # pred = self.lin(self.conv(x2)-self.conv(x1))
#         #     out1 = self.convs[kk](F.normalize(out) - F.normalize(x2[kk]))  # mix4
#         #     h = F.mean(out1, (2, 3))
#         #     # h = F.resize_images(out1,x1[-1].shape[2:])
#         #     feats.append(h)
#         # diff = F.concat(feats, 1)
#         # pred = self.lin(diff)
#         # return pred, diff
#
#         #     out1 = self.convs[kk](out) - self.convs(x2[kk]) # mix
#         #     # h = F.mean(out1, (2, 3))
#         #     # feats.append(F.resize_images(out1,x1[-1].shape[2:]))
#         #     feats.append(F.mean(out1, (2, 3)))
#         # diff = F.concat(feats, 1)
#         # pred = self.lin(diff)
#         # return pred, diff
#
#         #     out1 = self.convs[kk](F.normalize(out) - F.normalize(x2[kk]))
#         #     # out1 = self.convs[kk](F.concat((out, x2[kk]), 1))
#         #     # out1 = self.convs[kk](out) - self.convs[kk](x2[kk])
#         #     h = F.mean(out1, (2, 3))
#         #     # h = F.resize_images(out1, x1[-1].shape[2:])
#         #     feats.append(h)
#         # # diff = F.mean(F.concat(feats, 1), (2, 3))
#         # # pred = self.lin(self.conv(F.concat(feats, 1)))
#         # diff = F.concat(feats,1)
#         # pred = self.lin(diff)
#         # # pred = self.lin(self.conv(F.concat((x1[-1],x2[-1]),1)))
#
#
#         #     h = self.convs[kk](F.normalize(out))
#         #     # h = self.convs[kk](out)
#         #     feats.append(h)
#         #     h = F.mean(h, (2, 3))
#         #     # h = F.resize_images(out1, x1[-1].shape[2:])
#         #     hs.append(h)
#         # pred = self.lin(F.concat(hs,1))
#         # return pred, feats
#
#         #     h = F.normalize(out)       # attention
#         #     atte = self.convs[kk](h)
#         #     h = h*F.repeat(atte,h.shape[1],axis=1)
#         #     feats.append(h)
#         #     h = F.mean(h, (2, 3))
#         #     hs.append(h)
#         # pred = self.lin(F.concat(hs,1))
#         # return pred, feats
#
#
#         #     h = F.normalize(out)   #attention + cGan
#         #     atte = self.convas[kk](h)
#         #     h = self.convs[kk](h)
#         #     h = h*F.repeat(atte,h.shape[1],axis=1)
#         #     feats.append(h)
#         #     h = F.mean(h, (2, 3))
#         #     hs.append(h)
#         #     h2 = F.normalize(x2[kk])   #attention + cGan
#         #     atte2 = self.convas[kk](h2)
#         #     h2 = self.convs[kk](h2)
#         #     h2 = h2*F.repeat(atte2,h2.shape[1],axis=1)
#         #     h2 = F.mean(h2, (2, 3))
#         #     hs2.append(h2)
#         # val = hs[0] + hs[1] + hs[2] + hs[3] + hs[4]
#         # val2 = hs2[0] + hs2[1] + hs2[2] + hs2[3] + hs2[4]
#         # pred = self.lin(F.concat((val,val2),1))
#         # return pred, feats
#
#         #     h = F.normalize(out)   #attention + cGan-2
#         #     atte = self.convas[kk](h)
#         #     h = h*F.repeat(atte,h.shape[1],axis=1)
#         #     # h = self.convs[kk](h)
#         #     feats.append(h)
#         #     h2 = F.normalize(x2[kk])   #attention + cGan
#         #     atte2 = self.convas[kk](h2)
#         #     h2 = h2*F.repeat(atte2,h2.shape[1],axis=1)
#         #     h = self.convs[kk](h*h2)
#         #     d = F.mean(h, (2, 3))
#         #     hs.append(d)
#         # val = hs[0] + hs[1] + hs[2] + hs[3] + hs[4]
#         # pred = self.lin(val)
#         # return pred, feats
#
#         #     h = F.normalize(out)   #attention + cGan-2
#         #     atte = self.convas[kk](F.normalize(x2[kk]))
#         #     h = self.convs[kk](h)
#         #     # h = h*F.repeat(atte,h.shape[1],axis=1)
#         #     h = h*atte
#         #     # h = self.convs[kk](h)
#         #     feats.append(h)
#         #     # h = self.convs[kk](h)
#         #     d = F.mean(h, (2, 3))
#         #     hs.append(d)
#         # val = hs[0] + hs[1] + hs[2] + hs[3] + hs[4]
#         # pred = self.lin(val)
#         # return pred, feats
#
#
#             y = self.convas[kk](F.normalize(out))      #mul
#             y2 = self.convas[kk](F.normalize(x2[kk]))      #mul
#             feats.append(y)
#             # h = h*F.repeat(atte,h.shape[1],axis=1)
#             b, ch, h, w = y.data.shape
#             features = F.reshape(y, (b, ch, w * h))
#             features2 = F.reshape(y2, (b, ch, w * h))
#             gram = F.batch_matmul(features, features2, transb=True)
#             hs.append(gram)
#         # val = hs[0] + hs[1] + hs[2] + hs[3] + hs[4]
#         pred = self.lin(F.concat(hs,1))
#         return pred, feats
#
#         #     h1 = self.convs[kk](F.normalize(out))
#         #     h2 = self.convs[kk](F.normalize(x2[kk]))
#         # #     h1 = self.convs[kk](out)
#         # #     h2 = self.convs[kk](x2[kk])
#         #     # feats.append((h1,h2))
#         #     # feats.append(h1)
#         #     # h = F.resize_images(out1, x1[-1].shape[2:])
#         #     hs.append(F.mean(h1-h2, (2, 3)))
#         # pred = self.lin(F.concat(hs, 1))
#         # return pred, feats
#
#         # #     out1 = self.convs[kk](F.normalize(out))# - self.convs[kk](F.normalize(x2[kk]))  # mix
#         #     out1 = self.convs[kk](out)
#         #     # h = F.mean(out1, (2, 3))
#         #     # feats.append(F.resize_images(out1,x1[-1].shape[2:]))
#         #     feats.append(F.mean(out1, (2, 3)))
#         # diff = F.concat(feats, 1)
#         # pred = self.lin(diff)
#         # return pred, diff