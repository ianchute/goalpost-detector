import cv2
import numpy as np
import matplotlib.pyplot as plt

def cieluv(img, target):
    # Color distance in the CIELUV space
    # RGB space is not good for color detection
    img = img.astype('int')
    aR, aG, aB = img[:,:,0], img[:,:,1], img[:,:,2]
    bR, bG, bB = target
    rmean = ((aR + bR) / 2).astype('int')
    r2 = np.square(aR - bR)
    g2 = np.square(aG - bG)
    b2 = np.square(aB - bB)
    result = (((512+rmean)*r2)>>8) + 4*g2 + (((767-rmean)*b2)>>8)
    result = result.astype('float64')
    result -= result.min()
    result /= result.max()
    result *= 255
    result = result.astype('uint8')

    # skip sqrt for faster computation - just square the threshold instead
    return result

def compute_angle(x1,y1,x2,y2):
    if x2 == x1:
        return 0
    return np.arctan((y2-y1)/(x2-x1))

def compute_magnitude(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2. + (y2-y1)**2.)

def rt_degrees(rt):
    return np.array([[r, np.rad2deg(t)] for r, t in rt])

def find_lines(edges, threshold=65, minLineLength=40, maxLineGap=10):
    # probabilistic hough transform
    lines = cv2.HoughLinesP(
        edges, 
        1, 
        np.pi/180, 
        threshold, 
        minLineLength=minLineLength, 
        maxLineGap=maxLineGap
    )
    if lines is None:
        return []
    lines = [(compute_magnitude(*line[0]), compute_angle(*line[0])) for line in lines]
    return lines

def detect_goalpost(
    img, 

    grass_color=(100,150,50),
    grass_color_threshold=20,

    goalpost_color=(220,220,220),
    goalpost_color_threshold=5,
    
    median_kernel_size=5,
    morph_close_kernel_size=7,

    horizontal_line_threshold=10,
    vertical_line_threshold=10,

    debug=False
):
    # remove grass color
    grass = cieluv(img, grass_color) < grass_color_threshold
    img.flags.writeable = True
    img[grass] = [0,0,0]
    
    # extract goalpost color
    goalpost = cieluv(img, goalpost_color) < goalpost_color_threshold
    img_goalpost = goalpost.astype(bool).astype('uint8') * 255

    # clean goalpost image
    img_goalpost = cv2.medianBlur(img_goalpost, 5)
    kernel = np.ones((morph_close_kernel_size,morph_close_kernel_size),np.uint8)
    img_goalpost = cv2.morphologyEx(img_goalpost, cv2.MORPH_CLOSE, kernel)
    
    # compute image density
    density = img_goalpost.mean()/255

    if debug:
        print('density:', density)
        plt.imshow(img_color)
    else:
        if density < 0.3:
            # detect lines
            lines = find_lines(img_goalpost)
            lines = rt_degrees(lines)
        else:
            return False # [degenerate case] too many whites - goalpost could no longer be detected 

        # no lines found
        if len(lines) == 0:
            return False

        # an image has a goalpost if two perpendicular lines with 0 degrees and 90 degrees intersect
        horizontal_line = len(lines[(abs(lines[:,1]) < horizontal_line_threshold)]) > 0
        vertical_line = len(lines[(abs(abs(lines[:,1]) - 90) < vertical_line_threshold)]) > 0
        has_goalpost = horizontal_line and vertical_line
        return has_goalpost

if __name__ == '__main__':
    images = ['img/goalpost_true.jpg', 'img/goalpost_false.jpg']
    for img in images:
        print(img, detect_goalpost(plt.imread(img)))