#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import smdebug
import argparse
import json
import logging
import os
import sys
from torchvision.datasets import ImageFolder
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout)) 

import smdebug.pytorch as smd
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook


#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader, criterion, device,hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    logger.info("start testing....")
    
    hook.set_mode(smd.modes.EVAL)
    model.eval()
    test_loss = 0
    running_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output,target)
            pred = output.max(1, keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            running_loss += loss.item() * data.size(0)
            
            
        test_loss = running_loss /  len(test_loader.dataset)
        logger.info("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    test_loss, 
                    correct, 
                    len(test_loader.dataset), 
                    100.0 * correct / len(test_loader.dataset)
        ))
    

def train(model, train_loader,validate_loader, criterion, optimizer, device, args,hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
 
    for epoch in range(1, args.epochs + 1):
        logger.info("Start training....")
        
        hook.set_mode(smd.modes.TRAIN)
        model.train()
        running_loss = 0
        train_correct = 0 
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            running_loss += loss.item() 
            loss.backward()
            optimizer.step()
            pred = pred.argmax(dim = 1, keepdim= True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            
            
        logger.info("Start Validating.....")
        
        hook.set_mode(smd.modes.EVAL)
        model.eval()
        validate_loss = 0
        val_correct = 0
        with torch.no_grad():
            for data, target in validate_loader:
                data = data.to(device)
                target = target.to(device)
                pred = model(data)
                loss = criterion(pred, target)
                validate_loss += loss
                pred = pred.argmax(1, keepdim = True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()



        logger.info("Epoch : {} training_loss : {} training accuracy : {}% validating_loss : {} validating accuracy : {}%".format(
            epoch,
            running_loss / len(train_loader.dataset),
            (100 * (train_correct / len(train_loader.dataset))),
            validate_loss / len(validate_loader.dataset),
            (100 * (val_correct / len(validate_loader.dataset)))

        ))
        
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.require_grad = False
    
    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))

    return model

def create_data_loaders(data_dir, batch_size, mode):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    logger.info("Get {} data loader from s3 path {}".format(mode, data_dir))
    
    transformers = {
                    "training" :transforms.Compose([
                        transforms.RandomHorizontalFlip(p= 0.5),
                        transforms.Resize((200,200)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                    ]),
                    "testing":transforms.Compose([
                        transforms.Resize((200,200)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                    ]),
                    "validating":transforms.Compose([
                        transforms.Resize((200,200)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                    ])
                }
    #https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html
    data = ImageFolder(data_dir, transform = transformers[mode])
    data_loader = torch.utils.data.DataLoader(data,batch_size = batch_size, shuffle = True)
    return data_loader
    

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running on device {device}")
    model=net()
    model = model.to(device)
    
    
    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr = args.lr)
    
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)
    hook.register_loss(criterion)
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader = create_data_loaders(args.train , args.batch_size , "training")
    val_loader = create_data_loaders(args.valid , args.batch_size , "validating")

    model=train(model, train_loader,val_loader, criterion, optimizer, device, args,hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test_loader = create_data_loaders(args.test , args.test_batch_size , "testing")
    test(model, test_loader, criterion, device,hook)
    
    '''
    TODO: Save the trained model
    '''
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch-size",
        type = int ,
        default = 64, 
        metavar = "N",
        help = "input batch size for training (default : 64)"
    )
    parser.add_argument(
        "--test-batch-size",
        type = int ,
        default = 1000, 
        metavar = "N",
        help = "input test batch size for training (default : 1000)"
    )
    parser.add_argument(
        "--epochs",
        type = int ,
        default = 5, 
        metavar = "N",
        help = "number of epochs to train (default : 5)"
    )
    parser.add_argument(
        "--lr",
        type = float ,
        default = 0.001, 
        metavar = "LR",
        help = "learning rate (default : 0.001)"
    )
    
    
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--valid", type=str, default=os.environ["SM_CHANNEL_EVAL"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args=parser.parse_args()
    
    main(args)
