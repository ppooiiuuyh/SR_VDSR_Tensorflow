import argparse
import os
import pprint
import tensorflow as tf
from model import VDSR
import matplotlib.pyplot as plt
import numpy as np
from imresize import *

if __name__ == '__main__':
# =======================================================
# [global variables]
# =======================================================
    pp = pprint.PrettyPrinter()
    args = None
    DATA_PATH = "./train/"
    TEST_DATA_PATH = "./data/test/"
    
# =======================================================
# [add parser]
# =======================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_tag", type=str, default="VDSR tensorflow. Implemented by Dohyun Kim")
    parser.add_argument("--gpu", type=int, default=1) # -1 for CPU
    parser.add_argument("--epoch", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=41)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--num_innerlayer", type=int, default=18) #paper setup
    parser.add_argument("--lr_decay_rate", type=float, default=1e-1)
    parser.add_argument("--lr_step_size", type=int, default=20) #9999 for no decay
    parser.add_argument("--scale", type=int, default=3) #
    parser.add_argument("--checkpoint_dir", default="checkpoint")
    parser.add_argument("--cpkt_itr", default=0)  # -1 for latest, set 0 for training from scratch
    parser.add_argument("--save_period", type=int, default=1)
    parser.add_argument("--train_subdir", default="291")
    parser.add_argument("--test_subdir", default="Set5")
    parser.add_argument("--infer_subdir", default="Custom")
    parser.add_argument("--infer_imgpath", default="monarch.bmp") #monarch.bmp
    parser.add_argument("--type", default="eval", choices=["eval", "demo"]) #eval type uses .m data upscaled with matlab bicubic mathod on only Y chaanel, demo type uses raw images on RGB
    parser.add_argument("--c_dim", type=int, default=-1)  # 3 for RGB, 1 for Y chaanel of YCbCr
    parser.add_argument("--mode", default="train", choices=["train", "test", "inference", "test_plot"])
    parser.add_argument("--result_dir", default="result")
    parser.add_argument("--save_extension", default=".jpg", choices=["jpg", "png"])

    print("=====================================================================")
    args = parser.parse_args()
    if args.type == "eval" : args.c_dim = 1 ; args.train_subdir += "_M" ; args.test_subdir += "_M"
    elif args.type == "demo" : args.c_dim = 3 ;
    print("Eaxperiment tag : " + args.exp_tag)
    pp.pprint(args)
    print("=====================================================================")


# =======================================================
# [make directory]
# =======================================================
    if not os.path.exists(os.path.join(os.getcwd(),args.checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(),args.checkpoint_dir))
    if not os.path.exists(os.path.join(os.getcwd(),args.result_dir)):
        os.makedirs(os.path.join(os.getcwd(),args.result_dir))
        
        
# =======================================================
# [Main]
# =======================================================
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    config = tf.ConfigProto()
    if args.gpu == -1: config.device_count = {'GPU': 0}
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    #config.operation_timeout_in_ms=10000

    g = tf.Graph()
    g.as_default()
    with tf.Session(graph=g,config = config) as sess:
    # -----------------------------------
    # build model
    # -----------------------------------
        model_path = args.checkpoint_dir
        vdsr = VDSR(sess,args=args)

    # -----------------------------------
    # train, test, inferecnce
    # -----------------------------------
        if args.mode == "train":
            vdsr.train()


        elif args.mode == "test":
            vdsr.test()


        elif args.mode == "inference":
            #load image
            image_path = os.path.join(os.getcwd(),"test", args.infer_subdir, args.infer_imgpath)
            infer_image = plt.imread(image_path)
            if np.max(infer_image) > 1: infer_image = infer_image / 255
            infer_image = imresize(infer_image, scalar_scale=1, output_shape=None, mode="vec")

            sr_img = vdsr.inference(infer_image, depth = args.train_depth, scale = args.scale)
            plt.imshow(sr_img)
            plt.show()


        elif args.mode == "test_plot":
            #load image
            image_path = os.path.join(os.getcwd(),"test", args.infer_subdir, args.infer_imgpath)
            infer_image = plt.imread(image_path)
            if np.max(infer_image) > 1: infer_image = infer_image / 255
            infer_image = imresize(infer_image, scalar_scale=1, output_shape=None, mode="vec")

            vdsr.test_plot(infer_image)



