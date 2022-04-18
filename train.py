import torch
import torch.optim as optim
import argparse
from tqdm import tqdm
from dataset.coco_dataset import COCOImageDataset
import torchvision.transforms as transforms
import utils.utils as utils
from dataset.augmentation import *
from detectors.create_models import create_efficientdet_model, create_ssd_model
import time
import os

torch.manual_seed(0)
iteration = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=['ssd', 'efficientdet'], default="efficientdet", help="Detection model")
    parser.add_argument("--lr", default=0.0001, help="Learning rate")
    parser.add_argument("--num_epoch", default=100, help="Number of Epochs")
    parser.add_argument("--momentum", default=0.8, help="Learning momentum")
    parser.add_argument("--data", default="data/COCO-Hand/COCO-Hand-S", help="Location of dataset")
    parser.add_argument("--train_proportion", default=0.7, help="Proportion of data used for training")
    parser.add_argument("--batch_size", default=32, help="Batch size used for training")
    parser.add_argument("--num_workers", default=2, help="Number of workers for dataloading")
    parser.add_argument("--checkpoint_path", default="models/efficientdet/model.pth", help="path to checkpoint")
    parser.add_argument("--save_dir", default="models/efficientdet/")
    args = parser.parse_args()
    return args

def train_epoch(train_loader, model, scheduler, optimizer, epoch, args):
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
    global iteration
    print("{} epoch: \t start training....".format(epoch))
    start = time.time()
    total_loss = []
    model.train()
    optimizer.zero_grad()
    for idx, (images, annotations) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = images.cuda().float()
        annotations = annotations.cuda()
        classification_loss, regression_loss = model([images, annotations])
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        loss = regression_loss
        loss.backward()
      
        total_loss.append(loss.item())
        if(iteration % 300 == 0):
            print('{} iteration: training ...'.format(iteration))
            ans = {
                'epoch': epoch,
                'iteration': iteration,
                'cls_loss': classification_loss.item(),
                'reg_loss': regression_loss.item(),
                'mean_loss': np.mean(total_loss)
            }
            for key, value in ans.items():
                print('    {:15s}: {}'.format(str(key), value))
        iteration += 1
    scheduler.step(np.mean(total_loss))
    result = {
        'time': time.time() - start,
        'loss': np.mean(total_loss)
    }
    for key, value in result.items():
        print('    {:15s}: {}'.format(str(key), value))
    return result
    

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
    
    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"))
    model = model.to(device)
    model.to(device)

    # Loss and optimizer stuff
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, verbose=True)
    
    loss = 0.0
    for epoch in range(int(args.num_epoch)):
        # Train for 1 epoch
        model.train()
        loss = train_epoch(train_dataloader, model, scheduler, optimizer, epoch, args)

        # Evaluate
        model.eval()
        torch.save(model.state_dict(), os.path.join(os.path.abspath(args.save_dir), f"model.pth"))

    pass

if __name__ == "__main__":
    args = parse_args()
    main(args)