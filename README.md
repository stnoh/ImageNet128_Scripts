# Scripts for generating a CIFAR-style ImageNet128 dataset

## Introduction:
Based on the [code](https://github.com/PatrykChrabaszcz/Imagenet32_Scripts) for generating ImageNet32 and ImageNet64. This code has been optimized and you will be able to generate ImageNet128 on a machine with at least 64 GB RAM.

## Example
- Download and ectract the [dataset](http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads)
- Resize the images to 128x128:
  ```sh
  python3 image_resizer_imagenet.py -i ~/datasets/ImageNet/ILSVRC2012_img_train -o ~/datasets/ImageNet/ILSVRC2012_img_train_128
  python3 image_resizer_imagenet.py -i ~/datasets/ImageNet/ILSVRC2012_img_val -o ~/datasets/ImageNet/ILSVRC2012_img_val_128
  ```

- Convert the images to pickles:
  ```sh
  python3 image2numpy_imagenet_train.py  -i ~/datasets/ImageNet/ILSVRC2012_img_train_128/box -o ~/datasets/ImageNet128
  python3 image2numpy_imagenet_val.py  -i ~/datasets/ImageNet/ILSVRC2012_img_val_128/box -o ~/datasets/ImageNet128
  ```
- Load the dataset:
  ```py
  gen = imagenet128.load(MODE, BATCH_SIZE, data_dir=DATA_DIR)
    while True:
      for images, labels in gen():
        yield images, labels
  ```
