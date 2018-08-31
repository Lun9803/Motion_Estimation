import cv2 as cv
import numpy as np
import math
import sys

#Code by Bolun Li   25/08/2018


# setting up the input file
file_name = 'monkey'
file_format = '.avi'
demo_video = cv.VideoCapture(file_name+file_format)
# attributes of the file
width = int(demo_video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(demo_video.get(cv.CAP_PROP_FRAME_HEIGHT))
size = (width, height)
# motion estimation settings
frame_rate = demo_video.get(cv.CAP_PROP_FPS)
detect_period = 0.04
out_fps = int(1/detect_period)
frame_skipped = int(frame_rate * detect_period)
dot_colour = (255, 255, 255)
# width and length of each grid
r_w = int(width/150)
r_h = int(height/150)

detect_range = math.sqrt(math.pow(r_w/2, 2) + math.pow(r_h/2, 2))
valid_range = detect_range/4
colour_threshold = 2000
dot_thickness = 1
# setting up output file
out = cv.VideoWriter(file_name+'_out.avi', cv.VideoWriter_fourcc(*'XVID'), out_fps, size)

print("Start, configures: ")
print("height " + str(height) + " width: " + str(width))
print("height detect range: " + str(r_h) + " width detect range: " + str(r_w))
print("circular detect range: " + str(detect_range) + " colour threshold: " + str(colour_threshold))
print("detect every " + str(detect_period) + " seconds\n")


# limit the number n between min_n to max_n

def clamp(n, min_n, max_n):
    return max(min(max_n, n), min_n)


# calculate the absolute distance between two colours in two pixels

def diff(c1, c2):
    difference = 0
    for i in range(3):
        d = int(c1[i])-int(c2[i])
        difference += math.pow(d, 2)
    return difference


# find the difference of two blocks
# first block: in img1, centered by (h1, w1)
# second block: in img2, centered by (h2, w2)
# both block will have radius of r

def find_block_difference(img1, img2, h1, w1, h2, w2, r):
    start = -1*int(r/2)
    end = int(r/2)
    difference = 1
    for i in range(start, end+1):
        for j in range(start, end+1):
            colour1 = img1[clamp(h1+i, 0, height-1)][clamp(w1+j, 0, width-1)]
            colour2 = img2[clamp(h2+i, 0, height-1)][clamp(w2+j, 0, width-1)]
            difference += diff(colour1, colour2)
    return float(difference)/(r*r)


# find the most similar pixel in next frame within a given range(r)
# returns the coordinate of moved pixel (destination) and the difference of the block after movement

def find_vector(img1, img2, h, w, r):
    min_diff = sys.maxint
    destination = (h, w)
    difference = 0
    left = max(w-r, 0)
    right = min(w+r, img1.shape[1])
    top = max(h-r, 0)
    bot = min(h+r, img1.shape[0])

    # if the block changes only a little bit, don't do anything
    # to reduce errors and increase performance
    if find_block_difference(img1, img2, h, w, h, w, r) < 100:
        return destination, difference
    for width in range(left, right):
        for height in range(top, bot):
            difference = find_block_difference(img1, img2, h, w, height, width, r)
            if difference < min_diff:
                min_diff = difference
                destination = (height, width)
    return destination, difference


# start processing

count = 0
ret, frame = demo_video.read()
next_frame = None
next_ret = True
while True:
    for i in range(frame_skipped):
        next_ret, next_frame = demo_video.read()
        if not next_ret:
            break
    if not next_ret:
        break
    out_image = np.zeros((height, width, 3), np.uint8)

    # go through every pixel of the frame and the frame after
    # when there is a detection need to be done, it finds the most similar block in next frame
    # if the colour difference or distance of moved block reaches threshold, draw a white dot on the graph at that pixel
    # so the parts of the video that are moving will be highlighted with white dots
    for h in range(0, height):
        for w in range(0, width):
            if h % r_h == r_h/2 and w % r_w == r_w/2:
                destination, difference = find_vector(frame, next_frame, h, w, int(detect_range))
                distance = math.sqrt(math.pow((destination[0]-h), 2) + math.pow((destination[1]-w), 2))
                if distance >= valid_range and difference >= colour_threshold:
                    # print("h: " + str(h) + " w: " + str(w))
                    # print(destination)
                    # print("dis: " + str(distance) + " diff:  " + str(difference))
                    cv.line(out_image, (w, h), (w, h), dot_colour, thickness=dot_thickness)
                else:
                    out_image[h][w] = frame[h][w]
            else:
                out_image[h][w] = frame[h][w]
    out.write(out_image)
    count += 1
    print("frame " + str(count) + " done")
    frame = next_frame


out.release()
demo_video.release()
cv.destroyAllWindows()
print("Process finished")
