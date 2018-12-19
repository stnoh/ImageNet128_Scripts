# Scripts for generating a CIFAR-style ImageNet128 dataset

Code is based on:  
https://github.com/loshchil/SGDR   https://github.com/Lasagne/Recipes/blob/master/papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py  
https://gist.github.com/FlorianMuellerklein/3d9ba175038a3f2e7de3794fa303f1ee  


**Paper**:  
[A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets](https://arxiv.org/abs/1707.08819) by Patryk Chrabaszcz, Ilya Loshchilov and Frank Hutter 

**Webpage:**  
https://patrykchrabaszcz.github.io/Imagenet32/

**Dataset:**  
http://image-net.org/download-images

## Example
Resize the images to 128x128:
```sh
python3 image_resizer_imagenet.py -i ~/datasets/ImageNet/ILSVRC2012_img_train -o ~/datasets/ImageNet/ILSVRC2012_img_train_128
python3 image_resizer_imagenet.py -i ~/datasets/ImageNet/ILSVRC2012_img_val -o ~/datasets/ImageNet/ILSVRC2012_img_val_128
```

Convert the images to pickles:
```sh
python3 image2numpy_imagenet_train.py  -i ~/datasets/ImageNet/ILSVRC2012_img_train_128/box -o ~/datasets/ImageNet128
python3 image2numpy_imagenet_val.py  -i ~/datasets/ImageNet/ILSVRC2012_img_val_128/box -o ~/datasets/ImageNet128
```
Load the dataset:
```py
imagenet128.load(MODE, BATCH_SIZE, data_dir=DATA_DIR)
  while True:
    for images, labels in gen():
      yield 2. / 255 * images - 1, labels
```
