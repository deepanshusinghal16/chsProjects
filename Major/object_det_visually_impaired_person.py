import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 180) 
engine.setProperty('volume', 0.9)

import cv2
import numpy as np
import urllib.request
import time
stream_url = 'http://192.168.112.197/cam-mid.jpg'
print("Starting: ",stream_url)

whT = 320  
confThreshold = 0.5
nmsThreshold = 0.3  # overlap above 30% then only detection
classesfile = 'coco.names'
classNames = []

with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)  # Processing 
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # Backend
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Target system 

def findObject(outputs, img):  # Output  by backend of YOLO
    hT, wT, cT = img.shape  # height, width ,number of channels(color) from shape 
    bbox = [] 
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]  # By using scores = det[5:], the line extracts only the confidence scores for each class from the detection result.
            classId = np.argmax(scores)  # line determines the index (or class ID) in the scores array where the maximum confidence score occurs.
            confidence = scores[classId]  # represents the index of the class with the maximum confidence score, which was determined using np.argmax(scores)
            if confidence > confThreshold: 
                w, h = int(det[2]*wT), int(det[3]*hT)  # Yolo v3 provides box in a ratio
                x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)  # The (det[0], det[1]) represents the center coordinates of the bounding box
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence)) 

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)  # This code block is responsible for performing non-maximum suppression (NMS) and drawing bounding boxes with class labels on the detected objects.
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        object_name = classNames[classIds[i]]
        print("Detected Object:", object_name)
        engine.say(object_name + " Ahead")
        engine.runAndWait()
        
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)



while True:
    stream = None
    while stream is None:
        try:
            stream = urllib.request.urlopen(stream_url)
        except:
            print("Failed to open stream. Retrying in 5 seconds...")
            time.sleep(5)
            continue
    
    data = bytes()
    while True:
        data += stream.read(1024)  # Reads 1024 bytes of data from the stream and appends it to the data bytes object.
        a = data.find(b'\xff\xd8')  # Marks beginning and ending of jpeg image
        b = data.find(b'\xff\xd9')
        if a != -1 and b != -1:  # if both ending and starting of jpeg is found
            jpg = data[a:b+2]  # This line extracts a portion of the received data (data) corresponding to a complete JPEG image
            data = data[b+2:]  # It discards the portion of the data that has already been processed as a JPEG image.
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)  # This line decodes the extracted JPEG image data (jpg) into an OpenCV image format.
            break
    
    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)  # image preprocessing like adjusting width and height and dividing pixel by 255
    net.setInput(blob)  # This prepares the network to perform forward pass inference on the input image.
    outputNames = net.getUnconnectedOutLayersNames()  # Retrieves the names of the output layers of the neural network model.
    outputs = net.forward(outputNames)  # The outputs variable contains the raw output of the neural network, which includes the bounding box coordinates, objectness scores, and class scores for each detected object.
    findObject(outputs, img)  
    cv2.imshow('Image', img)  
    
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()  
