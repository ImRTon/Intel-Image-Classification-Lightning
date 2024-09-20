from lightning import LightningModule, LightningDataModule
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

import torch

class IntelImageClassificationDataModule(LightningDataModule):
    def __init__(
        self, 
        train_data_dir: str | Path, 
        val_data_dir: str | Path | None = None,
        split_ratio: float = 0.85,
        batch_size: int = 32,
        cls_map: dict | None = None,
        num_class: int | None = None,
        class_weights: list[float] | None = None,
    ):
        super().__init__()
        
        if isinstance(train_data_dir, str):
            train_data_dir = Path(train_data_dir)
        if isinstance(val_data_dir, str):
            val_data_dir = Path(val_data_dir)
            
        if cls_map is None:
            cls_map = {}
            
        self.datas, class_counts = self.read_data(train_data_dir, cls_map)
        num_class = len(cls_map)
        avg_count = sum(class_counts) / num_class
        class_weights = [max(0.1, min(10, avg_count / count)) for count in class_counts]
        self.val_datas = []        

        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.cls_map = cls_map
        self.num_class = num_class
        self.class_weights = class_weights

        self.save_hyperparameters()
            
        print(self.hparams.cls_map, self.hparams.num_class, self.hparams.class_weights)

    def read_data(self, data_dir: str | Path, cls_map: dict):
        datas = []
        class_counts = []
        for img_path in data_dir.rglob("*.jpg"):
            if img_path.parent == data_dir:
                # No label, it's predict dataset
                label = -1
            else:
                label = img_path.parent.name
                if label not in cls_map:
                    cls_map[label] = len(cls_map)
                    class_counts.append(0)

                label = cls_map[label]
                class_counts[label] += 1
                
            datas.append((img_path.relative_to(data_dir), label))
        return datas, class_counts

    def setup(self, stage=None):
        if stage == "fit":
            self.transform = self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),        
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),     
            ])
            self.val_transform = self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),            
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            ])
            
            if self.val_data_dir is None:
                self.datas, self.val_datas = train_test_split(self.datas, train_size=self.split_ratio)
            else:
                self.val_datas = self.read_data(self.val_data_dir)
                
        elif stage == "test":
            self.transform = self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),   
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),          
            ])
        elif stage == "predict":
            self.transform = self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),   
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),          
            ])

    def train_dataloader(self):
        return DataLoader(
            IntelImageClassificationDataset(
                self.datas, 
                self.hparams.train_data_dir, 
                self.transform
            ),
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=16,
        )

    def val_dataloader(self):
        return DataLoader(
            IntelImageClassificationDataset(
                self.val_datas, 
                self.hparams.train_data_dir, 
                self.val_transform
            ),
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=16,
        )

    def test_dataloader(self):
        return DataLoader(
            IntelImageClassificationDataset(
                self.datas, 
                self.hparams.train_data_dir, 
                self.transform
            ),
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=16,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            IntelImageClassificationDataset(
                self.datas, 
                self.hparams.train_data_dir, 
                self.transform
            ),
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=16,
        )
    
class IntelImageClassificationDataset(Dataset):
    def __init__(self, datas: list[tuple[Path, int]], root_dir: Path | str, transform=transforms.Compose):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
            
        self.root_dir = root_dir
        self.datas = datas
        self.transform = transform
        
    def __getitem__(self, index):
        img_path, label = self.datas[index]
        image = Image.open(self.root_dir / img_path).convert("RGB")
        image = self.transform(image)
        
        return image, torch.tensor(label)

    def __len__(self):
        return len(self.datas)