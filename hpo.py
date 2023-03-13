#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

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


def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("start testing....")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()* data.size(0)
            pred = output.max(1, keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        logger.info("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    test_loss, 
                    correct, 
                    len(test_loader.dataset), 
                    100.0 * correct / len(test_loader.dataset)
        ))
    

def train(model, train_loader, criterion, optimizer, device, args):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("Start training....")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0
        correct = 0 
        for batch_idx , (data, target) in enumerate(train_loader, 1):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss
            loss.backward()
            optimizer.step()
            output = output.argmax(dim = 1, keepdim= True)
            correct += output.eq(target.view_as(output)).sum().item()
            if batch_idx % 100 == 0:
                logger.info("Train Epoch : {} [{}/{}({:.0f}%)] loss:{:.6f} accuracy:{:.6f}%".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100* batch_idx/ len(train_loader),
                    loss.item(),
                    100*(correct/len(train_loader.dataset))
                ))
        
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    num_feature = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_feature, 133))
    return model

def create_data_loaders(data_dir, batch_size, mode):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    logger.info("Get {} data loader from s3 path {}".format(mode, data_dir))
    
    transformers = {
                    "training" : transforms.Compose([
                        transforms.RandomHorizontalFlip(p= 0.5),
                        transforms.Resize((200,200)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                    ]),
                    "testing":transforms.Compose([
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
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader = create_data_loaders(args.data_dir_train , args.batch_size , "training")
    
    model=train(model, train_loader, criterion, optimizer, device, args)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test_loader = create_data_loaders(args.data_dir_test , args.test_batch_size , "testing")
    test(model, test_loader, criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, "model.pth")
    logger.info(f"Saving the model in {path}.")
    torch.save(model, path)

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
    parser.add_argument("--data-dir-train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--data-dir-test", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--data-dir-valid", type=str, default=os.environ["SM_CHANNEL_VAL"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args=parser.parse_args()
    
    main(args)
