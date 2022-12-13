from PIL import Image, ImageOps
import numpy as np
import math
import glob

def get_gauss_val(mu, sigma, mins, maxs):
    ratio = np.random.normal(mu, sigma)
    ratio = max(mins, min(maxs, ratio))
    return ratio


def Gshape(a1,a2,b1,b2,c,sizes):
    #Z = np.zeros(sizes)
    h,w = sizes[0],sizes[1]
    x = [(i-a1)/b1 for i in range(0,h)]
    y = [(j-a2)/b2 for j in range(0,w)]
    x = np.array(x).reshape(h,1)
    y = np.array(y).reshape(1,w)
    Z = (x**2+y**2)+ 2*c*x*y

    #Z = -Z/(2*sigma)
    #Z = exp(Z) / (sqrt(2*pi) * sqrt(sigma))

    return np.exp(-Z)


def Gauss_mask(sizes):
    #mask1 = np.zeros((3,256,256))
    mask = np.zeros(sizes)
    h,w = mask.shape
    #print(mask)
    a1 = np.random.randint(0,sizes[0]-1)
    a2 = np.random.randint(0,sizes[1]-1)
    b1 = np.random.uniform(h/6,h/3)
    b2 = np.random.uniform(w/6,w/3)

    c = np.random.uniform(-0.5,0.5)
    #Rweight = np.random.normal(0.15,0.1,mask.shape)
    #Rweight[Rweight>0.3]=0.3
    #Rweight[Rweight<0]=0
    # print(v,w,h)
    mask = Gshape(a1,a2,b1,b2,c,mask.shape)#*Rweight
    return mask


def build_masks(sizes,num):# num=3
    mask_loader = np.zeros(sizes)# [H, W]
    norms = math.ceil(math.sqrt(sizes[0]*sizes[1])/384)# view it as a square, and compare the langth with 384
    gauss_num=int(norms)
    if num>1:
        gauss_num = np.random.randint(norms,norms*num)
    for j in range(0,gauss_num):
        mask_loader = mask_loader + Gauss_mask(sizes)
    # mask_loader[mask_loader>thres]=thres
    return mask_loader


def crop_imgs(img, label, crop_width=256):
    #impatch = np.ones((crop_width,crop_width))*255
    #lbpatch = np.ones((crop_width,crop_width))*255
    h,w = img.shape
    x = np.random.randint(0,h-crop_width)
    y = np.random.randint(0,w-crop_width)

    impatch = img[x:(x+crop_width),y:(y+crop_width)]
    lbpatch = label[x:(x+crop_width),y:(y+crop_width)]
    return impatch, lbpatch

# def loadimgs(dataDir, data_range, size):
#     dataset = []
#     for i in range(data_range[0],data_range[1]):
#         img = Image.open(dataDir+"/img-%04d.jpg"%i).convert('L')
#         label = Image.open(dataDir+"/label-%04d.jpg"%i).convert('L')
#         w,h = img.size
#         num = int(math.ceil(w /size)*math.ceil(h/size))
#
#         srnd = np.random.random() * 0.3 + 0.7
#         if int(srnd*w) > size and int(srnd*h) > size:
#             img = img.resize((int(img.size[0] * srnd), int(img.size[1] * srnd)), Image.ANTIALIAS)
#             label = label.resize((int(label.size[0] * srnd), int(label.size[1] * srnd)), Image.ANTIALIAS)
#         elif w <= size or h <= size:
#             scale = min(w,h)*1.0/(size+1)
#             img = img.resize((int(w / scale), int(h / scale)), Image.ANTIALIAS)
#             label = label.resize((int(w / scale), int(h / scale)), Image.ANTIALIAS)
#
#         img = ImageOps.autocontrast(img, 0)
#         img = np.asarray(img)
#         label = np.asarray(label)
#
#         masknum = 2
#         noisenum = 2
#         tonenum = 2
#
#         # tonechange
#         for i in range(0, tonenum):
#             tonernd = np.random.random() * 0.5
#             # shading
#             for i in range(0, masknum):
#                 mask = build_masks(img.shape)
#                 # noise
#                 for i in range(0, noisenum):
#                     Rweight = np.random.normal(0.15, 0.2, img.shape)
#                     Rweight[Rweight > 0.3] = 0.3
#                     Rweight[Rweight < 0] = 0
#                     img1 = img * (1 - tonernd) + 255 * tonernd
#                     img1 = img1 * (1 - mask * Rweight)
#                     img1 = img1.astype("f")/127.5-1.0
#                     label1 = label.astype("f")/127.5-1.0
#                     for i in range(0, num):
#                         img_, label_ = crop_imgs(img1, label1, size)
#                         dataset.append((img_,label_))
#                         dataset.append((np.flip(img_,1),np.flip(label_,1)))
#                         dataset.append((np.flip(img_,0),np.flip(label_,0)))
#                         dataset.append((np.flip(np.flip(img_,1),0),np.flip(np.flip(label_,1),0)))
#
#
#         # img1 = np.asarray(img)
#         # img2 = np.asarray(label)
#         #
#         # for i in range(0,2):
#         #     tonernd = np.random.random()*0.5
#         #     for i in range(0,2):
#         #         mask = build_masks(img1.shape)
#         #         for i in range(0,2):
#         #             Rweight = np.random.normal(0.2,0.3,img1.shape)
#         #             Rweight[Rweight>0.4]=0.4
#         #             Rweight[Rweight<0]=0
#         #
#         #             img_ = img1 * (1 - tonernd) + 255 * tonernd #img1*tonernd+(1-tonernd)/2*np.random.random()*255  #contrast
#         #             img_ = img_*(1-mask*Rweight).astype("f")/127.5-1.0
#         #             label_ = img2.astype("f")/127.5-1.0
#         #             for i in range(0,num):
#         #                 patch1, patch2 = crop_imgs(img_, label_, size)
#         #                 dataset.append((patch1,patch2))
#         #                 dataset.append((np.flip(patch1,1),np.flip(patch2,1)))
#         #                 dataset.append((np.flip(patch1,0),np.flip(patch2,0)))
#         #                 dataset.append((np.flip(np.flip(patch1,1),0),np.flip(np.flip(patch2,1),0)))
#
#     print("load dataset done")
#     print(len(dataset))
#     return dataset

def loadimgs(dataDir, size):
    dataset = []
    imgs = glob.glob(dataDir + '/img-*.jpg')
    labels = glob.glob(dataDir + '/label-*.jpg')
    # length = 176
    
    imgs.sort()
    labels.sort()
    # maxx = 0
    # minx = 1e7
    # sumx = 0
    for (imgp,labelp) in zip(imgs,labels):
        
        img = Image.open(imgp).convert('L')# [H, W]
        label = Image.open(labelp).convert('L')# [H, W]
        w,h = img.size
        # if w <= size or h <= size:
        #     scale = min(w, h) * 1.0 / (size + 1)
        #     img = img.resize((int(w / scale), int(h / scale)), Image.ANTIALIAS)
        #     label = label.resize((int(w / scale), int(h / scale)), Image.ANTIALIAS)
        #     w, h = img.size
        
        num = int(math.ceil(w /size)*math.ceil(h/size))
        # it means approximately, how many sub-imgs can be extracted from the img, without overlaps
        
        # if w*h>maxx:
        #     maxx = w*h
        # if w*h<minx:
        #     minx=w*h
        # sumx += w*h
        # if img.size != label.size:
        #     print(imgp)
        #     print(img.size)
        #     print(labelp)
        #     print(label.size)

        # at least half and a little bit more
        srnd = np.random.random() * 0.75 + 0.5
        while not (int(srnd*w) > size and int(srnd*h) > size):
            srnd = np.random.random() * 0.75 + 0.5
        if int(srnd*w) > size and int(srnd*h) > size:
            img = img.resize((int(img.size[0] * srnd), int(img.size[1] * srnd)), Image.ANTIALIAS)
            label = label.resize((int(label.size[0] * srnd), int(label.size[1] * srnd)), Image.ANTIALIAS)
        else:
            continue
        
        
        img = ImageOps.autocontrast(img, 0)
        img = np.asarray(img)
        label = np.asarray(label).astype("f")
        label[label>210]=255
        label[label<45]=0

        # img1 = img.astype("f") / 127.5 - 1.0
        # label1 = label.astype("f") / 127.5 - 1.0
        # for n in range(0, num):
        #     img_, label_ = crop_imgs(img1, label1, size)
        #     dataset.append((img_, label_))

        # masknum = 1
        # noisenum = 2
        #
        # # tonechange
        # # shading
        # for j in range(0, masknum):
        #     # mask = build_masks(img.shape, 3)
        #     # shadowmask = np.clip(mask, 0.35, 1)
        #     # tonemask = np.clip(mask - 0.5, 0, 0.5)
        #     # noise
        #     for m in range(0, noisenum):
        #         # img1=img
        #         Rweight = np.random.normal(1, 0.2, img.shape)
        #         Rweight = Rweight.clip(0.8, 1)
        #         img1 = img*Rweight
        #         # if np.random.random()>0.5:
        #         #     img1 = img1 * (1 - tonemask) + 255 * tonemask
        #         # if np.random.random()>0.5:
        #         #     # if np.random.random()>0.5:
        #         #     img1 = img * (1 - (1 - shadowmask) * Rweight)
        #         img1 = img1.astype("f") / 127.5 - 1.0
        #         label1 = label/ 127.5 - 1.0
        #         for n in range(0, num):
        #             img_, label_ = crop_imgs(img1, label1, size)
        #             dataset.append((img_, label_))
        #             dataset.append((np.flip(img_, 1), np.flip(label_, 1)))
        #             dataset.append((np.flip(img_, 0), np.flip(label_, 0)))
        #             dataset.append((np.flip(np.flip(img_, 1), 0), np.flip(np.flip(label_, 1), 0)))

        masknum = 2
        noisenum = 2

        # tonechange
        # shading
        
        for j in range(0, masknum):# 0, 1
            mask = build_masks(img.shape, 3)
            shadowmask = np.clip(mask, 0.35, 1)
            # clip function, to limit the ratio of mask, 0.35 to make it not be too dark, while 1.0 to make sure that it satisfies the value of a pixel
            tonemask = np.clip(mask - 0.55, 0, 0.45)
            # noise
            for m in range(0, noisenum):# two iterations
                img1=img# copy
                Rweight = np.random.normal(0.8, 0.25, img.shape)
                Rweight = Rweight.clip(0.6, 1)
                img1 = img*Rweight
                if np.random.random()>0.5:
                    img1 = img1 * (1 - tonemask) + 255 * tonemask
                if np.random.random()>0.5:
                    # if np.random.random()>0.5:
                    img1 = img1 * (1 - (1 - shadowmask) * Rweight)
                img1 = img1.astype("f") / 127.5 - 1.0
                label1 = label/ 127.5 - 1.0
                for n in range(0, num):
                    
                    img_, label_ = crop_imgs(img1, label1, size)
                    # label_[label_>=0.9]=1
                    # label_[label_<0.9]=0
                    dataset.append((img_, label_))
                    dataset.append((np.flip(img_, 1), np.flip(label_, 1)))
                    dataset.append((np.flip(img_, 0), np.flip(label_, 0)))
                    dataset.append((np.flip(np.flip(img_, 1), 0), np.flip(np.flip(label_, 1), 0)))


        # masknum = 2
        # noisenum = 2
        #
        # # tonechange
        # # shading
        # for j in range(0, masknum):
        #     mask = build_masks(img.shape, 2)
        #     shadowmask = np.clip(mask, 0, 0.5)
        #     tonemask = np.clip(1 - mask, 0, 0.5)
        #     # noise
        #     for m in range(0, noisenum):
        #         img1=img
        #         Rweight = np.random.normal(0.15, 0.2, img.shape)
        #         Rweight = Rweight.clip(0, 0.3)
        #         # img1 = img * (1 - Rweight)
        #         if np.random.random()>0.5:
        #             img1 = img1 * (1 - tonemask) + 255 * tonemask
        #         if np.random.random()>0.5:
        #             img1 = img1 * (1 - shadowmask) * (1 - Rweight * shadowmask)
        #         img1 = img1.astype("f") / 127.5 - 1.0
        #         label1 = label.astype("f") / 127.5 - 1.0
        #         for n in range(0, num):
        #             img_, label_ = crop_imgs(img1, label1, size)
        #             dataset.append((img_, label_))
        #             dataset.append((np.flip(img_, 1), np.flip(label_, 1)))
        #             dataset.append((np.flip(img_, 0), np.flip(label_, 0)))
        #             dataset.append((np.flip(np.flip(img_, 1), 0), np.flip(np.flip(label_, 1), 0)))

        # masknum = 1
        # noisenum = 2
        # tonenum = 1
        #
        # # tonechange
        # for i in range(0, tonenum):
        #     # tonemask = build_masks(img.shape,0.25)
        #     # shading
        #     for j in range(0, masknum):
        #         # mask = build_masks(img.shape,0.75)
        #         # noise
        #         for m in range(0, noisenum):
        #             # tonernd = np.random.random() * 0.5
        #             Rweight = np.random.normal(0.15, 0.2, img.shape)
        #             Rweight[Rweight > 0.3] = 0.3
        #             Rweight[Rweight < 0] = 0
        #             img1 = img*Rweight
        #             # img1 = img * (1 - tonemask) + 255 * tonemask
        #             # img1 = img * (1 - tonernd) + 255 * tonernd
        #             # img1 = img1 * (1 - Rweight)*(1 - mask)
        #             # img1 = img1 * (1 - mask * Rweight)
        #             img1 = img1.astype("f")/127.5-1.0
        #             label1 = label.astype("f")/127.5-1.0
        #             for n in range(0, num):
        #                 img_, label_ = crop_imgs(img1, label1, size)
        #                 dataset.append((img_,label_))
        #                 dataset.append((np.flip(img_,1),np.flip(label_,1)))
        #                 dataset.append((np.flip(img_,0),np.flip(label_,0)))
        #                 dataset.append((np.flip(np.flip(img_,1),0),np.flip(np.flip(label_,1),0)))


        # img1 = np.asarray(img)
        # img2 = np.asarray(label)
        #
        # for i in range(0,2):
        #     tonernd = np.random.random()*0.5
        #     for i in range(0,2):
        #         mask = build_masks(img1.shape)
        #         for i in range(0,2):
        #             Rweight = np.random.normal(0.2,0.3,img1.shape)
        #             Rweight[Rweight>0.4]=0.4
        #             Rweight[Rweight<0]=0
        #
        #             img_ = img1 * (1 - tonernd) + 255 * tonernd #img1*tonernd+(1-tonernd)/2*np.random.random()*255  #contrast
        #             img_ = img_*(1-mask*Rweight).astype("f")/127.5-1.0
        #             label_ = img2.astype("f")/127.5-1.0
        #             for i in range(0,num):
        #                 patch1, patch2 = crop_imgs(img_, label_, size)
        #                 dataset.append((patch1,patch2))
        #                 dataset.append((np.flip(patch1,1),np.flip(patch2,1)))
        #                 dataset.append((np.flip(patch1,0),np.flip(patch2,0)))
        #                 dataset.append((np.flip(np.flip(patch1,1),0),np.flip(np.flip(patch2,1),0)))
    # print(math.sqrt(minx))
    # print(math.sqrt(maxx))
    # print(math.sqrt(sumx/len(imgs)))

    print("load dataset done")
    print(len(dataset))
    return dataset


# img = np.asarray(Image.open('test2.jpg').convert('L')).astype('f')
# # tonemask = build_masks(img.shape,0.5,2)
# mask = build_masks(img.shape,3)#np.clip(build_masks(img.shape,2),0,1)
# Image.fromarray(np.uint8(mask*255)).save('mask.jpg')
# shadowmask = np.clip(mask,0.45,1)#np.clip(mask,0,np.random.random()*0.3+0.5)
# Image.fromarray(np.uint8(shadowmask*255)).save('shadowmask.jpg')
# # tonemask = 0.5*mask/mask.max()#np.clip(mask,np.random.random()*0.2+0.5,1)#np.clip(mask,0.5,1)
# # tonemask[tonemask<0.25]=0
# tonemask = np.clip(mask-0.35,0,0.65)
# Image.fromarray(np.uint8(tonemask*255)).save('tonemask2.jpg')
# Rweight = np.random.normal(1, 0.2, img.shape)
# Rweight = Rweight.clip(0.8,1)
# img1 = img * (1 - tonemask) + 255 * tonemask#255-(255-img) * tonemask
# Image.fromarray(np.uint8(img1)).save('testtone.jpg')
# img1 = img * shadowmask
# Image.fromarray(np.uint8(img1)).save('testshadow.jpg')
# img1 = img * Rweight
# Image.fromarray(np.uint8(img1)).save('testnoise.jpg')
# shadow = (1-(1-shadowmask)* Rweight)
# Image.fromarray(np.uint8(shadow*255)).save('shadownoise.jpg')
# # img1 = img * shadowmask* Rweight
# img1 = img * (1-(1-shadowmask)* Rweight)
# Image.fromarray(np.uint8(img1)).save('testshadownoise.jpg')
# img1 = img * (1 - tonemask) + 255 * tonemask#img * tonemask + 255 * (1 - tonemask)
# img1 = img1 * (1-(1-shadowmask)* Rweight)# * (1 - Rweight*shadowmask)* (1 - Rweight*tonemask)
# Image.fromarray(np.uint8(img1)).save('testshadownoisetone.jpg')