from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import v2
from TemplateDataloader import SexBaseDataset
import torch
import numpy as np
import random
from torchvision.transforms.v2.functional import InterpolationMode
from torchvision import tv_tensors

mean = [0.46617496, 0.36034706, 0.33016744]
std = [0.23478602, 0.21425594, 0.20241965]
num_epochs = 100
test_frequency = 10 #1 for debug, 10 for real world
LearningRate = 0.001
batch_size = 4

print (f"Going to train for {num_epochs} epoch")
class SexDataset(SexBaseDataset):
    def __init__(self, folder_path, transforms=None, augmented = False):
        super().__init__(folder_path)
        self.transforms = transforms
        self.augmented = augmented    
        if self.transforms is None:
            self.transforms = v2.Compose([    
                #v2.ToDtype(torch.float32, scale=True), #Auto scale from 0-255 to 0.0-1.0
                v2.Normalize(mean=mean, std=std)
            ])
        if self.augmented == True:
            self.augmentPosition = v2.Compose([                
                v2.RandomHorizontalFlip(0.5),
                v2.RandomVerticalFlip(0.15),
                v2.RandomRotation(degrees = 5),
                v2.RandomAffine(degrees = 5),
                v2.RandomPerspective(distortion_scale=0.15, p=0.80)
            ])
            self.augmentColour = v2.Compose([
                v2.ColorJitter(brightness=0.35, contrast = 0.25, saturation = 0.1, hue = 0.04),
                v2.GaussianNoise(mean = 0, sigma = 0.01) #default sigma 0.1
            ])
    # Override the __getitem__ method
    def __getitem__(self, idx):
        # Call the parent class's __getitem__ to get the default behavior
        image, mask = super().__getitem__(idx)
        #print (torch.min(mask),' ',torch.max(mask))
        image = v2.functional.to_dtype (image, scale = True)
        if self.augmented == True:
            image = self.augmentColour(tv_tensors.Image(image))
            image, mask = self.augmentPosition (tv_tensors.Image(image), tv_tensors.Mask(mask)) 
        if self.transforms is not None:
            image = self.transforms(image)
        mask = mask[0,:,:]/255
        #print (torch.min(mask),'_',torch.max(mask))
        return image, mask
        
#Dataset, dataloader
from torch.utils.data import DataLoader
import torch.optim as optim
Dataset = SexDataset ("G:\\jav folder\\OutputFolder", augmented = True)
train_loader = DataLoader(Dataset, batch_size=batch_size, shuffle=False)

#model, weight, training hyperparameter
weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights)
model.eval()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LearningRate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()
scaler = torch.amp.GradScaler('cuda')

#class to index of output array.
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}

def startTraining (num_epochs):
    best_loss = 1
    for epoch in range(num_epochs):
        running_loss = 0.0
        #display some first images and the prediction of models.
        for images, masks in train_loader:
            break
        if ((epoch%test_frequency) == 0) or (epoch < 15):
            for imgIdx in range (0, batch_size):
                #Original image
                fucking_image = ((images[imgIdx]*torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)))*255
                fucking_image = to_pil_image(fucking_image.to(torch.uint8))
                if epoch==0:
                    fucking_image.save("TrainPredict/Begin/" + str(imgIdx) + '_' + str(int(epoch)) + "_OG.png")
                    #Mask (Sanity check)
                    sexmask = to_pil_image(torch.cat([torch.zeros((2,*images[imgIdx].shape[1:])), (masks[imgIdx]).detach().cpu().unsqueeze(0)], dim=0).to(torch.uint8)*255)
                    #combine (Sanity check)
                    fucking_image.paste(sexmask, (0, 0), to_pil_image((masks[imgIdx]).detach().cpu().unsqueeze(0).to(torch.uint8)*150))
                    fucking_image.save("TrainPredict/Begin/" + str(imgIdx) + '_' + str(int(epoch)) + "_MASK.png")
                #Model Output
                model_output = model(images[imgIdx].unsqueeze(0).to(device))['out'].softmax(dim=1)[:,class_to_idx["person"],:,:].detach().cpu()
                model_image = to_pil_image(torch.cat([model_output, torch.zeros((2,*model_output.shape[1:]))], dim=0))
                #combine
                fucking_image.paste(model_image, (0, 0), to_pil_image(model_output[0]))
                fucking_image.save("TrainPredict/" + str(imgIdx) + '_' + str(int(epoch)) + ".png")
            
            globals().update(locals())
            
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            # Casts operations to mixed precision
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)['out']
                loss = criterion(outputs, masks.long()*class_to_idx["person"])
            # Scales the loss, and calls backward()
            # to create scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #print (loss.item(), end = ' _ ')
            running_loss += loss.item()
            globals().update(locals())

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
        if (running_loss/len(train_loader)) < best_loss:
            best_loss = running_loss
            torch.save (model, "models/bestmodel.pt")
    torch.cuda.empty_cache()
    torch.save (model, "models/fuckmodel.pt")
    globals().update(locals()) #STORE ALL VARIABLE TO GLOBAL, the stackoverflow answer hates this though...
    
startTraining(num_epochs)

