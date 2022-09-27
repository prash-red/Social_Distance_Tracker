#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jetson.inference
import jetson.utils
#import ipywidgets 
#from IPython.display import display
from jetcam.utils import bgr8_to_jpeg as col
import cv2
import math
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt


# In[2]:


def is_close(p1, p2, p3):
    dst = math.sqrt(p1 ** 2 + p2 ** 2 + p3**2)
    return dst 
def convertBack(x, y, w, h): 
    #================================================================
    # 2.Purpose : Converts center coordinates to rectangle coordinates
    #================================================================  
    """
    :param:
    x, y = midpoint of bbox
    w, h = width, height of the bbox
    
    :return:
    xmin, ymin, xmax, ymax
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


# In[3]:


import depthai as dai

stepSize = 0.05

newConfig = False

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
spatialLocationCalculator = pipeline.createSpatialLocationCalculator()
camRgb = pipeline.createColorCamera()
xoutRgb = pipeline.createXLinkOut()
camRgb.preview.link(xoutRgb.input)

xoutDepth = pipeline.createXLinkOut()
xoutSpatialData = pipeline.createXLinkOut()
xinSpatialCalcConfig = pipeline.createXLinkIn()

xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")
xoutRgb.setStreamName("rgb")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
camRgb.setPreviewSize(1280, 720)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

lrcheck = False
subpixel = False

stereo.initialConfig.setConfidenceThreshold(255)
stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)
# Config
topLeft = dai.Point2f(0.4, 0.4)
bottomRight = dai.Point2f(0.6, 0.6)

config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
config.roi = dai.Rect(topLeft, bottomRight)

spatialLocationCalculator.setWaitForConfigInput(False)
spatialLocationCalculator.initialConfig.addROI(config)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

device = dai.Device(pipeline)


# In[4]:


def get_depth(centroid_dict, n):
    
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
    
    for i in range(0, n):
        x,y,xmin, ymin, xmax, ymax,w,h = centroid_dict[i]
        x1 =int(x)-2
        y1 = int(y)-2
        x2=int(x) +2
        y2=int(y) +2
        
        topLeft = dai.Point2f(x1, y1)
        bottomRight = dai.Point2f(x2, y2)
        config.roi = dai.Rect(topLeft, bottomRight)
        cfg = dai.SpatialLocationCalculatorConfig()
        cfg.addROI(config)
        spatialCalcConfigInQueue.send(cfg)
        # Output queue will be used to get the depth frames from the outputs defined above


        color = (255, 255, 255)

        #inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived
        #depthFrame = inDepth.getFrame()
        #depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        #depthFrameColor = cv2.equalizeHist(depthFrameColor)
        #depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        spatialData = spatialCalcQueue.get().getSpatialLocations()
        n = 0
        xavg = 0
        yavg = 0
        zavg = 0
        for depthData in spatialData:
            #roi = depthData.config.roi
            #roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            #xmin = int(roi.topLeft().x)
            #ymin = int(roi.topLeft().y)
            #xmax = int(roi.bottomRight().x)
            #ymax = int(roi.bottomRight().y)

            depthMin = depthData.depthMin
            depthMax = depthData.depthMax

            '''fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymin + 20), fontType, 0.5, 255)
            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymin + 35), fontType, 0.5, 255)
            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymin + 50), fontType, 0.5, 255)'''
            xavg += int(depthData.spatialCoordinates.x)
            yavg += int(depthData.spatialCoordinates.y)
            zavg += int(depthData.spatialCoordinates.z)
            n += 1
            
        centroid_dict[i] = (int(x), int(y), xmin, ymin, xmax, ymax, xavg/n, yavg/n, zavg/n)
        print(i, int(x), int(y), int(w),int(h),int(xavg/n), int(yavg/n), int(zavg/n))
        '''img = jetson.utils.cudaFromNumpy(depthFrameColor)
        image_widget = ipywidgets.Image(format = 'jpeg')

        array = jetson.utils.cudaToNumpy(img)


            #print(depthData.spatialCoordinates.x,depthData.spatialCoordinates.y,depthData.spatialCoordinates.z)
        image_widget.value = col(array)
        display(image_widget)'''
    
    return(centroid_dict)


# In[5]:


def cvDrawBoxes(detections, img,distthresh):
    distance = 0
    """
    :param:
    detections = total detections in one frame
    img = image from detect_image method of darknet
    :return:
    img with bbox
    """
    #================================================================
    # 3.1 Purpose : Filter out Persons class from detections and get 
    #           bounding box centroid for each person detection.
    #================================================================
    if len(detections) > 0:  						# At least 1 detection in the image and check detection presence in a frame  
        centroid_dict = dict() 						# Function creates a dictionary and calls it centroid_dict
        objectId = 0								# We inialize a variable called ObjectId and set it to 0
        for detection in detections:				# In this if statement, we filter all the detections for persons only
            # Check for the only person name tag 
            name_tag = detection.ClassID
            if name_tag == 1:                
                x, y, w, h = detection.Center[0],                            detection.Center[1],                            detection.Width,                            detection.Height      	# Store the center points of the detections
                
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))   # Convert from center coordinates to rectangular coordinates, We use floats to ensure the precision of the BBox
                
                #deptx,depty,deptz = getdepth(objectId,int(x)-2, int(y)-2,int(x) +2, int(y) +2)
                #deptx,depty,deptz = getdepth(xmin, ymin, xmax, ymax)
                
                # Append center point of bbox for persons detected.
                centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax,w,h) # Create dictionary of tuple with 'objectId' as the index center points and bbox
                objectId += 1 #Increment the index for each detection    
                #'''
        centroid_dict = get_depth(centroid_dict, objectId)
    #=================================================================#
    
    #=================================================================
    # 3.2 Purpose : Determine which person bbox are close to each other
    #=================================================================            	
        red_zone_list = [] # List containing which Object id is in under threshold distance condition. 
        red_line_list = []
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2): # Get all the combinations of close detections, #List of multiple items - id1 1, points 2, 1,3
            dx, dy ,dz = p1[6] - p2[6], p1[7] - p2[7],p1[8] - p2[8]  	# Check the difference between centroid x: 0, y :1
            distance = is_close(dx, dy,dz) 			# Calculates the Euclidean distance
            #text = "distance: {} x1 {} x2 {} y1 {} y2 {} z1 {} z2 {}".format(str(int(distance)),str(p1[6]),str(p2[6]), str(p1[7]),str(p2[7]),str(p1[8]), str(p2[8]))
            text = "distance: {} x {} y {} z {} ".format(str(int(distance)),str(abs(p1[6] - p2[6])),str(abs( p1[7] - p2[7])), str(abs(p1[8] - p2[8])))
            #print('distance: ', str(int(distance)),str(p1[6]),str(p2[6]), str(p1[7]),str(p2[7]),str(p1[8]), str(p2[8]), str(abs(p1[6] - p2[6])),str(abs( p1[7] - p2[7])), str(abs(p1[8] - p2[8])))
            location = (10,25)												# Set the location of the displayed text
            cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)  # Display Text
            if distance <= distthresh:						# Set our social distance threshold - If they meet this condition then..
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)       #  Add Id to a list
                    red_line_list.append(p1[0:2])   #  Add points to the list
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)		# Same for the second id 
                    red_line_list.append(p2[0:2])
        
        for idx, box in centroid_dict.items():  # dict (1(key):red(value), 2 blue)  idx - key  box - value
            if idx in red_zone_list:   # if id is in red zone list
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2) # Create Red bounding boxes  #starting point, ending point size of 2
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2) # Create Green bounding boxes
		#=================================================================#

		#=================================================================
    	# 3.3 Purpose : Display Risk Analytics and Show Risk Indicators
    	#=================================================================        
        #text = "People at Risk: %s" % str(len(red_zone_list)) 			# Count People at Risk
        #'''
        '''
        if distance != 0:
            text = "distance: {} x1 {} x2 {} y1 {} y2 {} z1 {} z2 {}".format(str(int(distance)),str(p1[6]),str(p2[6]), str(p1[7]),str(p2[7]),str(p1[8]), str(p2[8]))
            location = (10,25)												# Set the location of the displayed text
            cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)  # Display Text'''

        for check in range(0, len(red_line_list)-1):					# Draw line between nearby bboxes iterate through redlist items
            start_point = red_line_list[check] 
            end_point = red_line_list[check+1]
            check_line_x = abs(end_point[0] - start_point[0])   		# Calculate the line coordinates for x  
            check_line_y = abs(end_point[1] - start_point[1])			# Calculate the line coordinates for y
            #if (check_line_x < distthresh) and (check_line_y < 25):				# If both are We check that the lines are below our threshold distance.
            cv2.line(img, start_point, end_point, (255, 0, 0), 2)   # Only above the threshold lines are displayed. 
        #=================================================================#
        #'''
        
    return img


# In[6]:


#import depthai as dai

# Create pipeline
#pipeline = dai.Pipeline()

# Define source and output
'''camRgb = pipeline.createColorCamera()
xoutRgb = pipeline.createXLinkOut()

xoutRgb.setStreamName("rgb")

# Properties
camRgb.setPreviewSize(1280, 720)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking
camRgb.preview.link(xoutRgb.input)

# Connect to device and start pipeline
#with dai.Device(pipeline) as device:'''



print('Connected cameras: ', device.getConnectedCameras())
# Print out usb speed
print('Usb speed: ', device.getUsbSpeed().name)

# Output queue will be used to get the rgb frames from the output defined above
qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)


inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived

 # Retrieve 'bgr' (opencv format) frame
#cv2.imshow("rgb", inRgb.getCvFrame())
'''pic = inRgb.getCvFrame()
img = jetson.utils.cudaFromNumpy(pic)
image_widget = ipywidgets.Image(format = 'jpeg')
        

array = jetson.utils.cudaToNumpy(img)
    
image_widget.value = col(array)
display(image_widget)'''


# In[7]:


net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

#camera = jetson.utils.videoSource("/dev/video0")      # '/dev/video0' for V4L2
dis = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file


# In[ ]:


while dis.IsStreaming():
    inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived

    # Retrieve 'bgr' (opencv format) frame
    #cv2.imshow("rgb", inRgb.getCvFrame())
    pic = inRgb.getCvFrame()
    img = jetson.utils.cudaFromNumpy(pic)
    
    detections = net.Detect(img)

    array = jetson.utils.cudaToNumpy(img)
    cvDrawBoxes(detections, array,300)
    #cv2.imshow('rgb',array)
    out = jetson.utils.cudaFromNumpy(array)
    #image_widget.value = col(array)
    dis.Render(out)
    dis.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))


# In[ ]:




