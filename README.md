# OpenCV - C++
Image processing using OpenCV.

## "Project Clock"
The goal of the project was to create a programme that would be able to tell the time based on a picture of a analogue clock.

### Programme framework plan:

1. Loading image
2. Pre-processing
    - contrast correction
    - gaussian blur
    - detecting the clock face (*HoughCircles*)
    - cropping the region of interest
4. Segmentation
    - treshholding
    - dilatation
    - erosion
    - bit inversion
6. Analysis
    - detecting contours
    - removing small or non-rectangular contours
    - identyfing the hands of the clock
    - measuring the absolute angle between hands of the clock and Y axis
    - calculating the time based on the angle data
8. Saving output image

### Effects of the program:
Input:

![Analysis 1](Clock/Pictures/1.jpg)

Output

![Analysis 1 OUT](Clock/Pictures/out_1.jpg)
