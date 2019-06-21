# Calculate mAP for object detection
mAP(mean average precision) is an important parameter to tell whether an object detection model is good. Here I represent a python code to calculate mAP with given boxes edge from your prediction and data. The prediction should be in shape=(-1,5) [probability, xmin, ymin, xmax, ymax]and data in shape=(-1,4) [xmin, ymin, xmax, ymax]. If there is any bug inside, please contact me. Thanks!
I didn't include probability(That a box is a real box) into account for now. Coming soon.
