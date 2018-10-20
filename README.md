# SRCNN-Tensorflow
Tensorflow implementation of Convolutional Neural Networks for super-resolution.

## Prerequisites
 * python 3.x
 * Tensorflow > 1.5
 * Scipy version > 0.18 ('mode' option from scipy.misc.imread function)
 * matplotlib
 * argparse

## Properties (what's different from reference code)
 * This code requires Tensorflow. This code was fully implemented based on Python 3 differently from the original.
 * This code use Adam optimizer instead of GradienDecententOptimizer differently from the original.
 * This code supports tensorboard summarization
 * This code supports data augmentation (scale, rotation and mirror flip)
 * This code supports custom dataset


## Usage
```
usage: main_vdsr.py [-h] [--epoch EPOCH] [--batch_size BATCH_SIZE]
                    [--image_size IMAGE_SIZE] [--label_size LABEL_SIZE]
                    [--base_lr BASE_LR] [--lr_decay_rate LR_DECAY_RATE]
                    [--lr_step_size LR_STEP_SIZE] [--c_dim C_DIM]
                    [--scale SCALE] [--stride STRIDE]
                    [--checkpoint_dir CHECKPOINT_DIR] [--cpkt_itr CPKT_ITR]
                    [--result_dir RESULT_DIR] [--train_subdir TRAIN_SUBDIR]
                    [--test_subdir TEST_SUBDIR] [--infer_subdir INFER_SUBDIR]
                    [--infer_imgpath INFER_IMGPATH]
                    [--mode {train,test,inference}]
                    [--save_extension {jpg,png}]


optional arguments:
  -h, --help            show this help message and exit
  --epoch EPOCH
  --batch_size BATCH_SIZE
  --image_size IMAGE_SIZE
  --label_size LABEL_SIZE
  --base_lr BASE_LR
  --lr_decay_rate LR_DECAY_RATE
  --lr_step_size LR_STEP_SIZE
  --c_dim C_DIM
  --scale SCALE
  --stride STRIDE
  --checkpoint_dir CHECKPOINT_DIR
  --cpkt_itr CPKT_ITR
  --result_dir RESULT_DIR
  --train_subdir TRAIN_SUBDIR
  --test_subdir TEST_SUBDIR
  --infer_subdir INFER_SUBDIR
  --infer_imgpath INFER_IMGPATH
  --mode {train,test,inference}
  --save_extension {jpg,png}

```

 * For training, `python main.py --mode train --check_itr 0` [set 0 for training from scratch, -1 for latest]
 * For testing, `python main.py --mode test`
 * For inference with cumstom dataset, `python main.py --mode inference --infer_imgpath 3.bmp` [result will be generated in ./result/inference]
 * For running tensorboard, `tensorboard --logdir=./board` then access localhost:6006 with your browser

## Result
<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/SR_VDSR--tensorflow/master/asset/3.bmp" width="400">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/SR_VDSR--tensorflow/master/asset/3.bmp190_scale3.jpg" width="400">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/SR_VDSR--tensorflow/master/asset/compare3.png" width="400">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/SR_SRCNN/master/asset/srcnn_result1.png" width="400">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/SR_SRCNN/master/asset/srcnn_result2.png" width="400">

</p>
<p align="center">
<img src="https://raw.githubusercontent.com/ppooiiuuyh/SR_SRCNN/master/asset/srcnn_result3.png" width="400">
</p>



## References
* [michalkoziarski/VDSR-Tensorflow](https://github.com/michalkoziarski/VDSR) : reference source code
* [VDSR](https://arxiv.org/pdf/1511.04587.pdf) : reference paper


## Author
Dohyun Kim

