# **Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. The Pipeline

The pipeline consists of 7 steps:
1. Turn the image into grayscale (no misteries here)
2. Find the edges of the image with the canny function. (A gaussian filter is applied before the canny funciton to avoid paying attention to noise.
3. Select the region of interrest ( If this was done before the canny phase we would have the edge of the region as a detected edge. If this was done after the next phase we would have a lot more lines that aren't part of the lanes to try to make sense of . So step 3 is the best place for this filter)
4. Join the edge points that seem to be part of the same line with the Hough transform. ( this is an incredably clever idea of transforming point space to line parameter space and then see which lines would "explain" the most points)
5. Seperate lines of the left lane from lines of the right line and from noise lines. 
6. Join the lines of each lane into a single line.
7. Plot these lines on the original image.

Steps 1 to 4 are well covered in the course classes. So I will speak more in depth of steps 5 and 6.

##### Step 5: seperating right lane lines from left lane lines

The first thing that can be said of the left and right lanes are that they have different line slopes. One of them has a positive line slope, the other a negative one. The slope of a line can be calculated as follows:

	 scope = (y1-y2)/(x1-x2)

Its important to note that in computer graphics, y coordinates grow downward. So a positive slope goes down. Taking this into account we can say that a positive slopped line is probably from the right lane and a negatively sloped line is probably from the left lane.

If the slope is too small (positive or negative) then its very close to a horizontal line. Unless the car is way off track, these lines are probably not part of either lane so we will filter them out.

##### Step 6: joining the lines of each lane into a single line.

A vary simple heuristic for joining the lines is to get left lane point that is farthest up and to the right and join it with the left lane point that is farthest to bottom and the left. This does an ok job of finding a line to represent the lanes.

A problem with this simple heuristic is that using the maximum or minimum of something is very suceptible to outliars. To try to make this algorithm more robost  we will not only look at the maximum or minimum but at the N largest and the N smallests. But we wanted to draw just one line! what should we do with all these points? Let's avarage them. BUT, taking the mean is also vary suceptible to outliars and we are trying to avoid this, so we will take the median instead.

Though very simple this algorithm does a resonable job on the test images and on the test video. The challange video is infact a challange. 

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potencial shortcoming is that if a line has a positive slope but is far of on the left side it will still be considered part of the right lane. Also we consider all lines we identify from each lane as equaly important, when infact we could argue that longer lines are more representatives of their lanes than short lines.

One very serious down side of this algorithm is that it assumes that the car is already resonably well positioned on the lane. If it is travaling on top of a lane, for example, it will still think that the left side of the lane is the left lane and the right side is the right lane. Or even worse, if the car is going sideways on the road it will find some random pattern and *make it* into a lane. A general lane finding algorithm should be able to call out when there are no lanes and when the lanes are missorientes (or actually the car is missoriented).


### 3. Suggest possible improvements to your pipeline

A potential improvement would be to use the line lenght as a weight to decide which lines are more representative of the lanes. take only the top N longest lines into consideration.

