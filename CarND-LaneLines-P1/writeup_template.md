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
7. Extend the lines to the end of the region of interest
8. Plot these lines on the original image.

Steps 1 to 4 are well covered in the course classes. So I will speak more in depth of steps 5, 6 and 7.

##### Step 5: seperating right lane lines from left lane lines

The first thing that can be said of the left and right lanes are that they have different line slopes. One of them has a positive line slope, the other a negative one. The slope of a line can be calculated as follows:

	 scope = (y1-y2)/(x1-x2)

Its important to note that in computer graphics, y coordinates grow downward. So a positive slope goes down. Taking this into account we can say that a positive slopped line is probably from the right lane and a negatively sloped line is probably from the left lane.

If the slope is too small (positive or negative) then its very close to a horizontal line. Unless the car is way off track, these lines are probably not part of either lane so we will filter them out.

##### Step 6: joining the lines of each lane into a single line.

When speaking of joining many points with a single line, the most strait forward thing to do is a linear regretion on the points. In a linear regression you try to find the line which minimizes the distance from each point, which is pretty much exactly what we neet.

There are a few drawbacks though. One is that the linear regression actualy minimizes the squared distances. This means that it trys very hard to fit the outliers. Another thing that adds to this fault is that positive slopes and negative slopes can still leave a lot of lines that don't actually belong in any of the lanes. 

To try to mitigate these problems we only fit the linear regression for the points of the 10 longest lines of each lane. Longer lines are more likely to be part of the lanes and more likely to continue between frames. 

##### Step 7: exenting the lines

To make sure our lines extend the entire interrest region we find the points where the line intersects with the edges of the image. This is done by first finding the equation of the image:
 	
	y = slope*x -slope*x1+y1
	
Then we find the y that makes the x = 0 (left edge), and then the y that makes the x = x_size (right edge). Then we plot a line between these two points. But, of course, this will draw a line outside the region of interest as well so we pass the image through the region filter one more time.

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline

One shortcomming of this pipeline is that, it doesn't do a great job at identifying what is not a part of the lane, if there is a log, or an animal or even a change in the asfalt color in the middle of the track this pipeple will assume that it is either part of the left lane or part of the right lane. This would drive the linear regression very of track, as can be seen in the challange video of the notebook.

Another shortcomming is that it assumes that the car is well positioned on the road. If the car is off track or it is perpandicular to the lanes it will most likely produce trash information. These may seem like extream situations that should never happen, but if you want a car to be completly autonomous it needs to be able to produce useful information in what ever situation it is in. At the very least ther should be another algorithm (or machine learing model) to tell the car when its safe to trust the information from this first algorithm.


### 3. Suggest possible improvements to your pipeline

A potential improvement could come by doing the regression with a the absolute error rather than the the squared error. This would penalise less the outliers, so it would be esier to ignore some extreme outliars. 
Another potential improvement is to look for the parrelel lines from either edge of the lane. This would help filter out long things that are not our lane, like a change in asfault color for example.

