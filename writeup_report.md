## Writeup - Report

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./report_images/1-origin-images.png
[image2]: ./report_images/2-images-feature.png
[image3]: ./report_images/sliding_windows.jpg
[image4]: ./report_images/4-images-drawboxs.png
[image5-1]: ./report_images/5-images-result.png
[image5-2]: ./report_images/5-images-result-2.png
[image6]: ./report_images/6-images-heatmap.png
[image7]: ./report_images/7-images-result.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]


The code for this step (extracted HOG features) is contained in the 5th code cell of the IPython notebook ，in the function `get_hog_features`    

here I try many parameter combination to find the best parameter for the code.

Here is an example of one of each of the `vehicle HOG image` and 'non-vehicle HOG image':

![alt text][image2]



#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and find the best HOG parameters is below

```
color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient =  9 # HOG orientations
pix_per_cell = 5 # HOG pixels per cell
cell_per_block = 3 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the images features, the images features combined color features with HOG features in 'extract_features()'

I trained the SVM, and the best test accuracy rate I got is 0.9661 

I save the model in 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I search the windows code is in `find_cars()`

the code is below:

```
# Crop the unnecessary area from the image
    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    ctrans_tosearch = img_tosearch

    # Compute the sliding window's parameters based on the scale
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    # Convert the image as grayscale
    img_gray = cv2.cvtColor(ctrans_tosearch, cv2.COLOR_BGR2GRAY)
    ctrans_tosearch = conv_color_space(ctrans_tosearch,color_space)

    # Define blocks and steps as above
    nxblocks = (img_gray.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (img_gray.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog_whole= get_hog_features(img_gray, orient, pix_per_cell, cell_per_block,feature_vec=False)

    heatbox = []
    find_box = []
    # Judged each window and compute the heat map
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_features = hog_whole[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell
```

here I used the the scale parameter 1.5 and the below images was showed some sample boxes not all choosed boxes.

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on "ALL" scales using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5-1]
![alt text][image5-2]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a test image ：

![alt text][image6]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My test program may fail at the railings，I think I can change the detection algorithm to improve the detection accuracy，

