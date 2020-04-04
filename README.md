## Results
Please check the results in xxxx

## Requirements
* Python 3.7.4, torch 1.4, torchvision 0.5
* `pip install -r requirements`
or
* `pip install torch torchvision sklearn matplotlib pandas skimage scikit-image yacs`

## Data preparation
* Follow [data.md](data.md) for instructions

## Configurations
Configurable options are defined in `configs/defaults.py`. Those options are firstly override by those in the input config file, e.g. `configs/yaml/res50.yaml`, and lastly override by those set in the tailing arguments sent via the command line, e.g. `OUTPUT_DIR ./checkpoints/res50_base` in `python train_transfer.py --config=configs/yaml/res50.yaml OUTPUT_DIR ./checkpoints/res50_base`.

## [Transfer learning] Training base network on topK (e.g. 20) classes from pre-trained Resnet
`python train_transfer.py --config=configs/yaml/res50.yaml DATALOADER.FP_DATASET.TOPK 20 TRAIN.NUM_CLASSES 20 OUTPUT_DIR ./checkpoints/res50_base `
- [configs/yaml/res50.yaml](configs/yaml/res50.yaml) as a sample template
- The training scripts keep log (`log.txt`) and checkpoints at `./checkpoints/res50_base`

## [Transfer learning] Fine-tuning base network on bottomK (e.g. 123) classes from some pre-trained model (e.g. `model_0012_val_0.8788.pth`)
`python train_transfer.py --config=configs/yaml/res50_finetune.yaml DATALOADER.FP_DATASET.TOPK -1 DATALOADER.FP_DATASET.BOTTOMK 123 TRAIN.NUM_CLASSES 123 --load_ckpt ./checkpoints/res50_base/model_0012_val_0.8788.pth OUTPUT_DIR "./checkpoints/res50_finetune"`

## [Transfer learning] Useful config options
* Set `MODEL.NORM_FEATURES` and `MODEL.NORM_PROTOTYPES` both to `True` allows the model to learn classifier in a hypersphere space, while the radius of the space is controlled by `MODEL.RADIUS_PROTOTYPES`

## [Few-shot learning] Training with k-way, n-shot, testing with k-way, n-shot
* `python train_fewshot.py --config=configs/yaml/res50_fewshot.yaml OUTPUT_DIR ./checkpoints/res50_fewshot_eucl_k2_n1 MODEL.PROTONET.DISTANCE_METRIC Euclidean TRAIN.K_WAY 2 TRAIN.N_SHOT 1 TEST.K_WAY 2 TEST.N_SHOT 1`

or **cosine** as the distance metric with varying radiuses (e.g. 4.)

* `python train_fewshot.py --config=configs/yaml/res50_fewshot.yaml OUTPUT_DIR ./checkpoints/res50_fewshot_cos_r1_k2_n1 MODEL.PROTONET.DISTANCE_METRIC Cosine MODEL.PROTONET.COSINE_RADIUS 4. TRAIN.K_WAY 2 TRAIN.N_SHOT 1 TEST.K_WAY 2 TEST.N_SHOT 1`

* Note that it is possible to set different `k` and `n` for meta-training and meta-testing, e.g. `TRAIN.K_WAY 10 TRAIN.N_SHOT 1 TEST.K_WAY 2 TEST.N_SHOT 1`
