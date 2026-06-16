from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from CLIP import clip
import torchvision.transforms as transforms
from torchvision import models
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import StepLR

""" model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")#ViT-B/32
model.cuda()
for para in model.parameters():
    para.requires_grad = False """


def tta_transforms(image, mode):
    """
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
    """
    image = image.cpu().detach().numpy()
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # rotate counterwise 90 degree
        out = np.rot90(image, k=1, axes=(2,3))
    elif mode == 2:
        # rotate 180 degree
        out = np.rot90(image, k=2, axes=(2,3))
    elif mode == 3:
        # rotate 270 degree
        out = np.rot90(image, k=3, axes=(2,3))
    else:
        raise Exception('Invalid choice of image transformation')
    out = torch.from_numpy(out.copy()).cuda()  #.view(1,3,256,256)
    return out

def tta_inverse_transforms(image, mode):
    """
    Performs data augmentation of the input image inverse
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - rotate counterwise 90 degree
                2 - rotate 180 degree
                3 - rotate 270 degree
    """
    image = image.cpu().detach().numpy()
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # rotate counterwise 90 degree
        out = np.rot90(image, k=3, axes=(2,3))
    elif mode == 2:
        # rotate 180 degree
        out = np.rot90(image, k=2, axes=(2,3))
    elif mode == 3:
        # rotate 270 degree
        out = np.rot90(image, k=1, axes=(2,3))
    else:
        raise Exception('Invalid choice of image transformation')
    out = torch.from_numpy(out.copy()).cuda()  #.view(1,3,256,256)
    return out



def update_ema_variables(ema_model, model_stu, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model_stu.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def center_crop(tensor, output_size):
    _, _, h, w = tensor.size()
    th, tw = output_size
    x1 = int((w - tw) / 2)
    y1 = int((h - th) / 2)
    return tensor[:, :, y1: y1 + th, x1: x1 + tw]

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x

class Clip_TTA(nn.Module):
    def __init__(self, deblur_model, optimizer, steps=1, episodic=False, mt_alpha=0.99, rst_m=0.1, ap=0.9):
        super().__init__()
        self.model = deblur_model
        self.episodic = episodic
        self.optimizer = optimizer
        self.scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor, self.model_int = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.mt = mt_alpha
        self.rst = rst_m
        self.ap = ap
        self.weight_decay = 0.001
        """ self.clip_model, self.preprocess = open_clip.create_model_from_pretrained(
            "daclip_ViT-B-32", './clip_model/daclip_ViT-B-32.pt'
        )
        self.clip_model = self.clip_model.cuda() """

        self.clip_model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")#ViT-B/32
        self.clip_model = self.clip_model.cuda()

    def forward(self, x):
        torch.cuda.empty_cache()
        for i in range(self.steps):
            outputs = self.forward_adapt_network(x, self.optimizer, self.scheduler)
            
        return outputs
    
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_adapt_network(self, x, optimizer, scheduler):
        flag = 0
        image = x
        img_resize = (1120,1120)
        if(x.shape[2] > 1680 or x.shape[3] > 1150):
            #print("out")  
            x = center_crop(x, img_resize)
            flag = 1 
        
        with torch.no_grad():
            output_int = self.model_int(x)
     
        resize = transforms.Resize((224,224))
        output_int_resize = resize(output_int)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_context = self.clip_model.encode_image(
                output_int_resize#, control=True
            )
            #print(image_context[1].shape)
            image_context = image_context.float()

        outputs_pres = []
        with torch.no_grad():
            for i in range(4):
                torch.cuda.empty_cache()
                outputs_transfom_quan  = self.model_ema(tta_transforms(x,i),image_context) #
                outputs_itransfom_quan = tta_inverse_transforms(outputs_transfom_quan,i)
                outputs_pres.append(outputs_itransfom_quan) 
        
        mean = torch.stack(outputs_pres).mean(0)
        squared_diff = [(tensor - mean)**2 for tensor in outputs_pres]
        variance = torch.stack(squared_diff).mean(0)
        
        conf = 1 - torch.sigmoid(variance/0.001)
        #print("conf",conf)

        outputs_ema = mean

        for name, param in self.model.named_parameters():
            param.requires_grad_(True)
        
        output = self.model(x,image_context)   #
    
        """ img_PIL_output = TF.to_pil_image(output[0])
        img_PIL_output.save("output"+".png")

        img_PIL_outputs_ema = TF.to_pil_image(outputs_ema[0])
        img_PIL_outputs_ema.save("outputs_ema"+".png") """
        

        # Student update
        l1_A_loss = F.l1_loss(outputs_ema, output) #+ 0.5*F.l1_loss(output_int, output)
        #print("l1_A_loss",l1_A_loss)
        #if True:#l1_A_loss < 0.011:
        loss = l1_A_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        # Teacher update
        self.model_ema = update_ema_variables(self.model_ema, self.model, alpha_teacher=self.mt)
        # Stochastic restore
        if True:
            for nm, m  in self.model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<self.rst).float().cuda() 
                        with torch.no_grad():
                            p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1.-mask)
        

        if flag:
            outputs_pres_new = []
            with torch.no_grad():
                for i in range(4):
                    torch.cuda.empty_cache()
                    outputs_transfom_quan  = self.model_ema(tta_transforms(image,i),image_context ) #            
                    outputs_itransfom_quan = tta_inverse_transforms(outputs_transfom_quan,i)
                    img_PIL_output = TF.to_pil_image(outputs_transfom_quan[0])
                    #img_PIL_output.save("transfom"+".png")
                    outputs_pres_new.append(outputs_itransfom_quan) 
            outputs_ema = torch.stack(outputs_pres_new).mean(0)
            #img_PIL_output = TF.to_pil_image(outputs_ema[0])
            #img_PIL_output.save("zuizhhong"+".png")
        
        return outputs_ema

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if 'attention'  in nm:#: collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)

    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    print("Reload model and initialize teacher/student with identical weights.")
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    model_int = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    for param in model_int.parameters():
        param.detach_()   
    return model_state, optimizer_state, ema_model, model_anchor, model_int


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    # train mode, but no grad
    model.train()
    model.requires_grad_(False)

    for nm, m in model.named_modules():
        if 'attention' in nm:
            m.requires_grad_(True)
            print(nm)
    return model

def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class PerceptualLoss(nn.Module):
    def __init__(self, layers):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features
        self.layers = layers
        self.criterion = nn.MSELoss()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def forward(self, x, y):
        x_features = self.vgg(self.transform(x))
        y_features = self.vgg(self.transform(y))
        loss = 0
        for layer in self.layers:
            x_feat = x_features[layer]
            y_feat = y_features[layer]
            loss += self.criterion(x_feat, y_feat)
        return loss
    

class vgg_features(nn.Module):
  def __init__(self):
    super(vgg_features, self).__init__()
    # get vgg16 features up to conv 4_3
    self.model = nn.Sequential(*list(models.vgg16(pretrained=True).features)[:23])
    # self.model = self.model.cuda(1)
    self.model = self.model.cuda()
    # will not need to compute gradients
    for param in self.parameters():
      param.requires_grad = False

  def forward(self, x, renormalize=True):
    # change normaliztion form [-1,1] to VGG normalization
    if renormalize:
      x = ((x * .5 + .5) - torch.cuda.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1,
                                          1)) / torch.cuda.FloatTensor(
        [0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = x.cuda()
    x = self.model(x)
    return x   