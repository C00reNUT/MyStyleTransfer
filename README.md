# My Style Transfer

After seing the works of Google's DeepDream and Lucid, I wanted to make my own style transfer algorithm from scratch using Tensorflow and the Keras VGG19 pre-trained CNN classifier. 

## Dependencies
```
tensorflow vs. 2.0 (needs to be tf v2 or else it won't work)
numpy
pickle (included)
os (included)
```

## How to run

Edit the content_path and style)path inside "MainStyleTransfer.py" and run it. You will be asked to provide the number of iteration as well as if you want to restore. 

I suggest 1000 epochs, which can take anywhere from 1 hour for 360X480 pictures to 15 hours for a 1920X1080 image.

## Restrictions

This program is not for compute weaklings!!
Anything print-worthy (larger than 1920X1080) requires at least 32 gigs of ram, with 64 gigs being optimal. It is highly reccomended that you run this on a dataserver with CUDA acceleration because your computer will be incredibly slow when run this. 

The images MUST be the same size, and there is an assertion to prevent the wrong sizes. 

You can't restore a pickle image if there isn't one, and it is HIGHLY reccomended that you DON'T restore if you are using a new set of images (although I am really curious as to what would happen)

Change the content and style layers at your own exploration. 

## Examples
Right now, there is an image in the style and content each labeled "test.jpg", and the output should be labeled "combined.jpg". If you decide to use this repo, delete these and enter your own. 

## Dataflow and Theory

The three images: Style (s), Content (C) and generated image (G) are the main players here. 

The big idea is to optimize a tensor variable G such that the content loss and style loss is minimized. 

Content loss is derived by comparing CNN feature representations of C and G, and style loss is derived by comparing the Gram Matrix representations of each CNN layer from S and G. 

S and C get converted to feature maps at the start of training, and each iteration, the loss function is calculated and a gradient is established. Adamoptimizer with a very aggressive learning rate is applied and G is optimized. 

Every certain number of steps, G is saved to the "evolution" folder so you can view your image becoming really awesome. 