'''
	ML Group 8
	The goal of this portion of the project is focused on detecting edges of the road.
	This may be extended to lane detection, rather than just road edgeline detection.
	The ability to detect where the edges of the road are or where one's lane is, is an
	important component of autonomous driving. It is the core of having a car drive on its
	own.
'''
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os

# Constants needed
FILENAME = 'YellowUnderShade.jpg'

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
        cmap = 'gray' if len(image.shape)==2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        
    # show the plot containing images
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()
    

'''
	Loads in the passed in image to be used in the program
'''
def img_load(impath):
    # load in the image
    img = plt.imread(impath)
    return img
	

'''
	Transforms the image by selecting for white and yellow lane
	lines. Then, the image is converted to grayscale and smoothed
	with gaussian blur. The result is then returned
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
    
    # now increase constrast by converting to grayscale and smoothing edges
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
    return cv.Canny(img, 50, 150) # TODO: Change the lower and upper threshold values as needed

'''
	Select the region of where to look for edges given a set of
	vertecies. Returns an image with all other noise or detected edges
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
	
	# detect the straight lines 
    lines = cv.HoughLinesP(edge_img, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap = 300)
    
    # copy the image and draw lines on it
    out_img = img
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(out_img, (x1, y1), (x2, y2), [255, 0, 0], 2)
            
    return out_img
    
'''
	Main function to drive the program. Makes all of the needed function calls
	and sets up the needed params to run the program
'''
def main():
    # load image
    img = img_load(FILENAME)
	
	# transform the image
    transformed_img = prep_img(img)
	
	# detect the edges
    edges = edge_detect(transformed_img)
    
    
    # select a good region
    rows, cols = edges.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * .6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    
    cropped_lanes = select_lane_region(edges, vertices)
    
    # find and draw the lines
    output = find_draw_edges(cropped_lanes, img)
           
	# show the image
    images = [output]
    show_images(images)
    
main()
