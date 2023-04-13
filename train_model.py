#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
import argparse
from PIL import ImageFile
import importlib

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(model, test_loader, criterion, device, hook, args):
    smd = importlib.import_module('smdebug')
    modes = getattr(smd, 'modes')
    
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    hook.set_mode(modes.EVAL)
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs, labels = inputs.to(device), labels.to(device)
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    print(f"Testing output for Hyperparameters: epoch: {args.epochs}, lr: {args.lr}, batch size: {args.batch_size}, momentum: {args.momentum}")
    print(f"Testing Loss: {total_loss}, Testing Accuracy: {100*total_acc}")
    

def train(model, train_loader, validation_loader, criterion, optimizer, device, epochs, hook, args):
    
    smd = importlib.import_module('smdebug')
    modes = getattr(smd, 'modes')
    
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    print(f"Starting training for Hyperparameters: epoch: {args.epochs}, lr: {args.lr}, batch size: {args.batch_size}, momentum: {args.momentum}")
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                hook.set_mode(modes.TRAIN)
                model.train()
            else:
                hook.set_mode(modes.EVAL)
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                # if running_samples>(0.2*len(image_dataset[phase].dataset)):
                #     break

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

def model_fn(model_dir):
    model = net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model


def save_model(model, model_dir):
    print(f"Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.ImageFolder(data, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,num_workers=6,
            shuffle=True)

    return data_loader

def main(args):
    
    pytorch = importlib.import_module('smdebug.pytorch')
    get_hook= getattr(pytorch, 'get_hook')

    train_loader = create_data_loaders(os.environ['SM_CHANNEL_TRAIN'], args.batch_size)
    validation_loader = create_data_loaders(os.environ['SM_CHANNEL_VALID'], args.batch_size)
    test_loader = create_data_loaders(os.environ['SM_CHANNEL_TEST'], args.test_batch_size)
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum)

    '''if args.gpu == 1:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model.to(device)'''
    
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Training on device: {device}")
    
    hook = pytorch.Hook.create_from_json_file()
    #get_hook(create_if_not_exists=True)
    hook.register_hook(model)
    hook.register_loss(loss_criterion)
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, device, args.epochs, hook, args)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device, hook, args)
    
    '''
    TODO: Save the trained model
    '''
    save_model(model, args.model_dir)

if __name__=='__main__':

    profiler = importlib.import_module('smdebug.profiler.utils')
    str2bool = getattr(profiler, 'str2bool')

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
        default=5,
        metavar="N",
        help="number of epochs to train (default: 2)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--gpu", type=str2bool, default=True)
    
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="N", help="momentum"
    )
    # Container environment
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
        
    args=parser.parse_args()
    
    main(args)
