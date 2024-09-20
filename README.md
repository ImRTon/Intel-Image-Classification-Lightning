# Simple Intel Image Classification Implementation Using PyTorch Lightning
A simple implementation of ResNet50 image classification on Kaggle Intel Image Classification.  

## Training
```bash
python train.py fit --data.train_data_dir datasets/seg_train/ --data.batch_size 128 --model.lr 1e-2 --trainer.max_epochs 20
```

## Testing
```bash
python train.py test --data.train_data_dir datasets/seg_train/ --data.batch_size 128 --model.lr 1e-2 --ckpt_path lightning_logs/version_6/checkpoints/epoch\=19-step\=1880.ckpt
```

## Result

|        Test metric        |    Result                 |
|---------------------------|---------------------------|
|         Accuracy          |    0.9346666932106018     |
|         F1 Score          |    0.4783767759799957     |
