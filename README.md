###Tumor segmentation

###Run on multigpu

###Pytorch - torchrun

###Monai

###Model architecture - UNet - Pocketnet 

###To run

python main.py --gpus 2 --folderpath /path/to/input/output/files --lr le-3 --optimizer SGD

###Inputfiles 
- dataset.csv - path/to/data/in/csv/format
- input.json - number of channels, epochs, patchs, modality, mask, roisize

###Outputfiles

-index, log, model_weights, pred - New folders will be created in the folderpath

