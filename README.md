# Goalpost detector
*By: Ian Chu Te*

## Prerequisites
- Python 3
- numpy
- OpenCV
- matplotlib (for debugging)

## Installation
Download `goalpost_detector.py` or clone this repo

## Try it
`python goalpost_detector.py`

## Sample usage
```python
from goalpost_detector import detect_goalpost

images = ['img/goalpost_true.jpg', 'img/goalpost_false.jpg']
for img in images:
    print(img, detect_goalpost(plt.imread(img)))
```

## CV Techniques used
- Color thresholding (CIELUV space)
- Closing Morpohological Transform 
- Probabilistic Hough Transform