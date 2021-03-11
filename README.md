# PASCAL VOC DADASET GENERATION

This tools will convert large-scale satellite or drone images into the Pascal VOC 2012 Dataset format.

## Folder Structure

```Shell
 VOC2012
   ├── JPEGImages
   ├── SegmentationClass
   ├── ImageSets
       ├── Segmentation
           ├── trainaug.txt
           ├── val.txt
```

## Usage

```Shell
usage: main.py [-h] [--im-path IM_PATH] [--lb-path LB_PATH]
               [--split-size SPLIT_SIZE] [--padding] [--stride STRIDE]
               [--output-format {.png, .jpg}] [--color-mode {rgb, grayscale}]
               [--train-val-split] [--split-ratio SPLIT_RATIO]
               [--background-filter] [--background-ratio BACKGROUND_RATIO]
```
