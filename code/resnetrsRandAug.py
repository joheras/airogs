#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastai.vision.all import *
import fastai
import timm
import albumentations as A

# In[2]:


import torch
torch.cuda.set_device(2)

def randAugment(N, M, p, mode="all", cut_out = False):
    # Magnitude(M) search space  
    shift_x = np.linspace(0,150,10)
    shift_y = np.linspace(0,150,10)
    rot = np.linspace(0,30,10)
    shear = np.linspace(0,10,10)
    sola = np.linspace(0,256,10)
    post = [4,4,5,5,6,6,7,7,8,8]
    cont = [np.linspace(-0.8,-0.1,10),np.linspace(0.1,2,10)]
    bright = np.linspace(0.1,0.7,10)
    shar = np.linspace(0.1,0.9,10)
    cut = np.linspace(0,60,10)
    # Transformation search space
    Aug =[#0 - geometrical
        A.ShiftScaleRotate(shift_limit_x=shift_x[M], rotate_limit=0,   shift_limit_y=0, shift_limit=shift_x[M], p=p),
        A.ShiftScaleRotate(shift_limit_y=shift_y[M], rotate_limit=0, shift_limit_x=0, shift_limit=shift_y[M], p=p),
        A.Affine(rotate=rot[M], p=p),
        A.Affine(shear=shear[M], p=p),
        A.InvertImg(p=p),
        #5 - Color Based
        A.Equalize(p=p),
        A.Solarize(threshold=sola[M], p=p),
        A.Posterize(num_bits=post[M], p=p),
        A.RandomContrast(limit=[cont[0][M], cont[1][M]], p=p),
        A.RandomBrightness(limit=bright[M], p=p),
        A.Sharpen(alpha=shar[M], lightness=shar[M], p=p)]
    # Sampling from the Transformation search space
    if mode == "geo": 
        ops = np.random.choice(Aug[0:5], N)
    elif mode == "color": 
        ops = np.random.choice(Aug[5:], N)
    else:
        ops = np.random.choice(Aug, N)

    if cut_out:
        ops.append(A.Cutout(num_holes=8, max_h_size=int(cut[M]),   max_w_size=int(cut[M]), p=p))
    transforms = A.Compose(ops)
    return transforms, ops


class AlbumentationsTransform(DisplayedTransform):
    split_idx,order=0,2
    def __init__(self, train_aug): store_attr()
    
    def encodes(self, img: PILImage):
        aug_img = self.train_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)

# In[3]:




# In[4]:


dfValid = pd.read_csv('../dataset/labels.csv')


# In[5]:




# In[6]:
dfValid = dfValid[(dfValid.set == 'training') | (dfValid.set == 'validation')]



def is_valid(path):
    name = path[0]
    return (dfValid[dfValid['challenge_id']==name])['set'].values[0]=='validation'
# In[13]:

def get_class(path):
    name = path[0]
    return (dfValid[dfValid['challenge_id']==name])['class']



def get_dls(presize=224,size=128,M=0,bs=128):
    db = DataBlock(blocks = (ImageBlock, CategoryBlock),
             splitter=FuncSplitter(is_valid),
             get_x = ColReader(0,pref="../dataset/preprocess/",suff='.jpg'),
             get_y=ColReader(1),
             item_tfms = [Resize(presize),AlbumentationsTransform(randAugment(5,M,0.5)[0])], # CropPad(200,200)
             batch_tfms=[*aug_transforms(size=size, min_scale=0.75,do_flip=True,flip_vert=True,
                  max_rotate=2.,max_zoom=1.1, max_warp=0.05,p_affine=0.9, p_lighting=0.8), 
                         Normalize.from_stats(*imagenet_stats)])
    newdf = pd.concat([dfValid[(dfValid.set == 'validation')],
               dfValid[(dfValid.set == 'training') & (dfValid['class'] == 'RG')],
               dfValid[(dfValid.set == 'training') & (dfValid['class'] == 'NRG')].sample(2354)])
    dls = db.dataloaders(newdf.values,bs=bs)
    return dls


class ProgressiveResizingRandAugment(TrackerCallback):
    
    def __init__(self,increaseSizeEpochs = [0,5], presizes=[224,384],sizes=[128,224],Ms=[0,5],bs=256,nameModel='test'):
        super().__init__()
        self.presizes = presizes
        self.sizes = sizes
        self.increaseSizeEpochs = increaseSizeEpochs
        self.Ms = Ms
        self.bs = bs
        self.increments = -1
        self.nameModel = nameModel
        
    def after_epoch(self):
        super().after_epoch()
        for i,epochs in enumerate(self.increaseSizeEpochs):
            if(self.epoch==epochs):
                self.increments = i
                if (i!=0):
                    self.learn.load(self.nameModel)
        
        self.learn.dls = get_dls(self.presizes[self.increments],self.sizes[self.increments],
                                 self.Ms[self.increments],int(self.bs/(2**self.increments)))


from fastai.vision.all import *
callbacks = [
    ShowGraphCallback(),
    #EarlyStoppingCallback(patience=5),
    SaveModelCallback(fname='resnetrsRandAug',monitor='cohen_kappa_score'),
    ReduceLROnPlateau(patience=2), ProgressiveResizingRandAugment([0,100,200,300,350],[224,384,512,640,640],[128,224,384,512,512],[0,2,4,6,8],128,'resnetrsRandAug')
]


dlsTrain3 = get_dls()


learn = Learner(dlsTrain3,timm.create_model('resnetrs50',num_classes=2,pretrained=True),
                metrics=[accuracy,CohenKappa(weights='quadratic')],cbs=callbacks,
                loss_func= FocalLossFlat()).to_fp16()

learn.fine_tune(400,base_lr=1e-4)

