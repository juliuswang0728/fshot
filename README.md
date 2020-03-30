# fshot

# Data preparation
* Follow [data.md](data.md) for instructions

# Training base network from pre-trained Resnet
`python train_transfer.py --config=configs/yaml/res50.yaml OUTPUT_DIR ./checkpoints/res50_base`
- [configs/yaml/res50.yaml](configs/yaml/res50.yaml) as a sample template
- The training scripts keep log (`log.txt`) and checkpoints at `./checkpoints/res50_base`

# Configurations
The configurable options are defined in `configs/defaults.py`. Those options are firstly override by those in the input config file, e.g. `configs/yaml/res50.yaml`, and lastly override by those set in arguments in the command line, e.g. `python train_transfer.py --config=configs/yaml/res50.yaml OUTPUT_DIR ./checkpoints/res50_base`.
