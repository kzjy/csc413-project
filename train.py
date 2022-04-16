import torch
import torch.optim as optim
import argparse
from tqdm import tqdm
from dataset.coco_dataset import COCOImageDataset
import torchvision.transforms as transforms
import utils.utils as utils
from dataset.augmentation import *
from detectors.create_models import create_efficientdet_model, create_ssd_model

torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=['ssd', 'efficientdet'], default="efficientdet", help="Detection model")
    parser.add_argument("--lr", default=0.01, help="Learning rate")
    parser.add_argument("--num_epoch", default=100, help="Number of Epochs")
    parser.add_argument("--momentum", default=0.8, help="Learning momentum")
    parser.add_argument("--data", default="data/COCO-Hand/COCO-Hand-S", help="Location of dataset")
    parser.add_argument("--train_proportion", default=0.7, help="Proportion of data used for training")
    parser.add_argument("--batch_size", default=32, help="Batch size used for training")
    parser.add_argument("--num_workers", default=2, help="Number of workers for dataloading")
    args = parser.parse_args()
    return args

def train_epoch(model, criterion, dataloader, optimizer, device) -> float:
    """ 
    Train for 1 epoch on the dataset

    args: 
    model: Full model with backbone + detector
    criterion: loss function
    dataloader: dataloader
    optimizer: gradient optimizer
    device: device to run on

    returns:
    loss: float
    """
    epoch_loss = 0.0
    for i, (images, annotations) in tqdm(enumerate(dataloader), total=len(dataloader)):
        images = images.cuda().float()
        annotations = annotations.cuda()
        classification_loss, regression_loss = model([images, annotations])
        # print(outputs.size())
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        loss = classification_loss + regression_loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    return epoch_loss
    

def main(args):
    
    # Dataset stuff
    dataset = COCOImageDataset(args.data, transform=transforms.Compose(
            [Normalizer(), Augmenter(), Resizer()]))
    train_size = int(len(dataset) * float(args.train_proportion))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])

    # Dataloader stuff
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                    batch_size=int(args.batch_size),
                                                    shuffle=True, 
                                                    num_workers=int(args.num_workers),
                                                    collate_fn=collater)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, 
                                                    batch_size=int(args.batch_size),
                                                    shuffle=True, 
                                                    num_workers=int(args.num_workers),
                                                    collate_fn=collater)


    # Model stuff
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model == "efficientdet":
        model = create_efficientdet_model()
    else:
        model = create_ssd_model()
    model.to(device)

    # Loss and optimizer stuff
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=float(args.lr), momentum=float(args.momentum))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    loss = 0.0
    for epoch in range(int(args.num_epoch)):
        # Train for 1 epoch
        model.train()
        loss += train_epoch(model, criterion, train_dataloader, optimizer, device)
        lr_scheduler.step()

        # Evaluate
        model.eval()

    pass

if __name__ == "__main__":
    args = parse_args()
    main(args)