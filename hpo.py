#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import logging
import os
import sys

from PIL import ImageFile
import argparse


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device, args):
    '''
     This function takes a model and a 
          testing data loader and will get the test accuray/loss of the model
    '''
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    print(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")
    

def train(model, train_loader, validation_loader, criterion, optimizer, device, args):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    epochs=2
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    print(f"Starting training for Hyperparameters: epoch: {args.epochs}, lr: {args.lr}, batch size: {args.batch_size}, momentum: {args.momentum}")
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                if running_samples % 2000  == 0:
                    accuracy = running_corrects/running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )
                
                #NOTE: Comment lines below to train and test on whole dataset
                if running_samples>(0.2*len(image_dataset[phase].dataset)):
                    break

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1

        if loss_counter==1:
            break
    return model
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 2054), nn.ReLU(inplace=True), nn.Linear(2054, 128), nn.ReLU(inplace=True), nn.Linear(128, 133))
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.ImageFolder(data, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,num_workers=6,
            shuffle=True)

    return data_loader

def main(args):
    train_loader = create_data_loaders(os.environ['SM_CHANNEL_TRAIN'], args.batch_size)
    test_loader = create_data_loaders(os.environ['SM_CHANNEL_TEST'], args.test_batch_size)
    validation_loader = create_data_loaders(os.environ['SM_CHANNEL_VALID'], args.batch_size)
    
    '''
    Initializing the model
    '''
    model=net()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    '''
    loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum)
    

    
    '''
    Calling the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, device, args)
    
    
    
    '''
    Testgin the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device, args)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, 'resnet50_dogImages_classification_hyperpara.pt')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="N", help="momentum"
    )
        
    args=parser.parse_args()
    
    main(args)