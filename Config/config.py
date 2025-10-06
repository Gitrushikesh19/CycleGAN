import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.checkpoint import CheckpointPolicy, checkpoint

Device = 'cuda' if torch.cuda.is_available() else 'cpu'
Train_dir = 'data/train'
Val_dir = 'data/val'
Batch_size = 1
LR = 1e-5
Lambda_identity = 0.0
Lambda_cycle = 10
Num_epochs = 10
Num_workers = 1
Load_model = False
Save_model = True
Checkpoint_gen_z = 'genz.pth.tar'
Checkpoint_gen_h = 'genh.pth.tar'
Checkpoint_critic_z = 'criticz.pth.tar'
Checkpoint_critic_h = 'critich.pth.tar'

transforms = A.Compose([
    A.Resize(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5, 0.5, 0.5], max_pixel_value=255),
    ToTensorV2()
], additional_targets={'image':'image0'}
)
