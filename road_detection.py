'''
	ML Group 8
	The goal of this portion of the project is focused on detecting edges of the road.
	This may be extended to lane detection, rather than just road edge line detection.
	The ability to detect where the edges of the road are or where one's lane is, is an
	important component of autonomous driving. It is the core of having a car drive on its
	own.
'''
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os

# Constants needed
FILENAMES = ['YellowUnderShade.jpg', 'YellowWhite.jpg'] # list of photos to load

'''
	Function to display the images of the road. Can be called before
	or after the edge lines have been created. This takes in an array
	of images so that multiple pictures can be analyzed and shown all
	at once.
'''
def show_images(images, cmap=None):
    # plot setup
    cols = 2
    rows = (len(images)+1)//cols
    
    plt.figure(figsize=(10, 11))
    
    # plot each image in the passed in array
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i+1)
        # use gray scale color map if there is only one channel
        cmap = 'gray' if len(image.shape) == 2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        
    # show the plot containing images
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()
    
'''
	Transforms the image by selecting for white and yellow lane
	lines. Then, the image is converted to gray scale and smoothed
	with Gaussian blur. The result is then returned
'''
def prep_img(img):
    # transformations of the image to make edge lines pop
    # first select for yellow and white colors
    transformed_img = img
    
    # convert to hsl to make white and yellow pop in shade and sun
    transformed_img = cv.cvtColor(transformed_img, cv.COLOR_RGB2HLS)
    
    # white mask
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv.inRange(transformed_img, lower, upper)
    
    # yellow mask
    lower = np.uint8([10, 0, 100])
    upper = np.uint8([40, 255, 255])
    yellow_mask = cv.inRange(transformed_img, lower, upper)
    
    # combine the masks
    mask = cv.bitwise_or(white_mask, yellow_mask)
    transformed_img = cv.bitwise_and(transformed_img, transformed_img, mask=mask)
    
    # now increase contrast by converting to gray scale and smoothing edges
    cv.cvtColor(transformed_img, cv.COLOR_RGB2GRAY)
    cv.GaussianBlur(transformed_img, (17, 17), 0) # kernel may be changed, 17 gives a good smoothing
    
    # return the image
    return transformed_img

''' 
	Uses canny edge detection to find the edges of an image and returns the
	resulting detected edges
'''
def edge_detect(img):
    # edge line detection
    return cv.Canny(img, 50, 150)

'''
	Select the region of where to look for edges given a set of
	vertices. Returns an image with all other noise or detected edges
	outside of that region removed.
'''
def select_lane_region(img, vertices):
    mask = np.zeros_like(img)
    
    # remove the unneeded lines, taking into account a channel dimension
    if len(mask.shape) == 2:
        cv.fillPoly(mask, vertices, 255)
    else:
	    cv.fillPoly(mask, vertices, (255,) * mask.shape[2])

    return cv.bitwise_and(img, mask)


'''
	Function to find the edge lines on the transformed image and draw the lane lines
	on the original image for output. There will most likely be multiple lines for each
	lane detected, and therefore, the average slope of these lines is taken to give one,
	single output line.
'''
def find_draw_edges(edge_img, img):
    # arrays needed to weight and average line slopes
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []

    # copy the image and draw lines on it
    out_img = img
    
    # detect the straight lines and build lists to average
    lines = cv.HoughLinesP(edge_img, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap = 300)
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            if (x2 == x1) or (y2 == y1):    # vertical/horizontal line
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            if (abs(slope) < .4): # catch bad slopes/misread edges
                continue
            if slope < 0:   # left hand line
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:           # right hand line
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    # get the y coords for the lines
    y1 = img.shape[0]
    y2 = y1 * 0.65
    
    # make longer lines count more
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None 
        
    # get the points for the lines
    def get_line_points(y1, y2, line):
        if line is None:
            return None
        slope, intercept = line
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        y1 = int(y1)
        y2 = int(y2)
        
        return ((x1, y1), (x2, y2))
    
    # get the point representation of the lines
    left_lane = get_line_points(y1, y2, left_lane)
    right_lane = get_line_points(y1, y2, right_lane)
    
    lines = (left_lane, right_lane)
    
    # actually draw the lines on the image
    line_img = np.zeros_like(img)
    for line in lines:
        if line is not None:
            cv.line(line_img, *line, [169, 69, 255], 20)
            
    return cv.addWeighted(img, 1.0, line_img, .95, 0.0)
    
'''
	Main function to drive the program. Makes all of the needed function calls
	and sets up the needed params to run the program
'''
def detect_lanes(imgs):
	
	# transform the image
    transformed_imgs = [prep_img(img) for img in imgs]
    
	# detect the edges
    edges = [edge_detect(img) for img in transformed_imgs]
    
    # select a good region
    cropped_lanes = []
    for img in edges:
        rows, cols = img.shape[:2]
        bottom_left = [cols * 0.1, rows * 0.95]
        top_left = [cols * 0.4, rows * .6]
        bottom_right = [cols * 0.9, rows * 0.95]
        top_right = [cols * 0.6, rows * 0.6]
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        cropped_lanes.append(select_lane_region(img, vertices))
    
    # find and draw the lines
    outputs = [find_draw_edges(cropped, img) for cropped, img in zip(cropped_lanes, imgs)]
           
