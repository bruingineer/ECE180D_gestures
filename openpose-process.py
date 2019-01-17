# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import numpy as np 
import paho.mqtt.client as mqtt

DEBUG_MQTT = False
DEBUG_MAIN = False
DEBUG_PROCESS_KEYPOINTS = False

# numpy suppress sci notation, set 1 decimal place
np.set_printoptions(suppress=True)
np.set_printoptions(precision=1)


waiting_for_target = True


# mqtt setup
ip = "131.179.28.219"
port = 1883

CONNECTED = False
client = None 
target_topic = 'gesture'
return_topic = 'gesture_correct'
target_gesture = "stop"

def on_connect(client, userdata, flags, rc):
    print("Connected with rc: "+str(rc))
    CONNECTED = True
    if DEBUG_MQTT:
        print("target_gesture: "+target_gesture)


def on_message(client, userdata, msg):
    print("msp received: "+msg.topic+" "+str(msg.payload))
    if msg.topic == target_topic:
        global target_gesture
        global waiting_for_target
        target_gesture = str(msg.payload)
        
        if DEBUG_MQTT:
            print("MQTT.on_message: message from "+msp.topic+"\ntarget_gesture="+target_gesture)

        if target_gesture == "stop":
            waiting_for_target = True
        else:
            waiting_for_target = False
            if DEBUG_MQTT:
                print("set waiting to "+waiting_for_target+". gesture = "+target_gesture)


def connect_to_server(ip, port):
    global client
    client = mqtt.Client(client_id = 'openpose')
    client.on_connect = on_connect
    client.on_message = on_message
    print("connect_to_server: target_gesture = "+target_gesture)
    client.connect(ip, port, 60)
    client.subscribe(target_topic, qos=0)

connect_to_server(ip, port)
client.loop_start()

# Remember to add your installation path here
# Adds directory of THIS script to OS PATH (to search for necessary DLLs & models)
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, dir_path + "/")

# Parameters for OpenPose. Take a look at C++ OpenPose example for meaning of components. Ensure all below are filled
try:
    from openpose import *
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')

params = dict()
params["logging_level"] = 3
params["output_resolution"] = "-1x-1"
params["net_resolution"] = "-1x368"
params["model_pose"] = "BODY_25"
params["alpha_pose"] = 0.6
params["scale_gap"] = 0.3
params["scale_number"] = 1 
params["render_threshold"] = 0.05
# If GPU version is built, and multiple GPUs are available, set the ID here
params["num_gpu_start"] = 0
params["disable_blending"] = False
# Ensure you point to the correct path where models are located
params["default_model_folder"] = dir_path + "/models/"

# Construct OpenPose object allocates GPU memory
try:
    openpose = OpenPose(params)
except:
    raise Exception('Error: OpenPose object could not be made with given params.')
print("OpenPose object created.")

# reference for coordinates given in keypoints np array
# (0,0)-------->(WIDTH,0)
#   |               |
#   |               |
# (0,HEIGHT)--(WIDTH,HEIGHT)

# keypoints shape [people x parts x index] == (1L, 25L, 3L)
# index[] = [x , y , confidence]

# keypoints for BODY_25
body25 = {
    "Nose":      0,
    "Neck":      1,
    "RShoulder": 2,
    "RElbow":    3,
    "RWrist":    4,
    "LShoulder": 5,
    "LElbow":    6,
    "LWrist":    7,
    "MidHip":    8,
    "RHip":      9,
    "RKnee":    10,
    "RAnkle":   11,
    "LHip":     12,
    "LKnee":    13,
    "LAnkle":   14,
    "REye":     15,
    "LEye":     16,
    "REar":     17,
    "LEar":     18,
    "LBigToe":  19,
    "LSmallToe":20,
    "LHeel":    21,
    "RBigToe":  22,
    "RSmallToe":23,
    "RHeel":    24,
    "Background":25
}

class ProcessKeyPoints:
    def __init__(self, keypoints, inputWIDTH, inputHEIGHT):
        self.keypoints = np.array(keypoints)
        self.num_people = self.keypoints.shape[1]
        self.WIDTH = inputWIDTH
        self.HEIGHT = inputHEIGHT

    def checkFor(self, _target_gesture):
        if _target_gesture == "tpose":
            return self.isTPose()
        elif _target_gesture == "fieldgoal":
            return self.isFieldGoal()
        else:
            print("gesture "+_target_gesture+" not recognized.")

    # checks for arms straight out from sides, using passed in keypoints to class
    # returns true if in TPose
    def isTPose(self):
        # 1D array: y of [body25 1 thru 7]
        # self.TPoseKeypoints_y = self.keypoints[0,1:8,1][ self.keypoints[0,1:8,1].flat > 0 ]
        # OR
        parts = [body25[x] for x in ['Neck','RShoulder','RElbow','RWrist','LWrist','LElbow','LShoulder']]
        a = self.keypoints[0, parts, 1].flat
        self.TPoseKeypoints_y = a[a > 0]

        # print(self.TPoseKeypoints_y)
        # print(self.TPoseKeypoints_y1)
        # print("")

        if (self.TPoseKeypoints_y.size >= 6) and (((self.TPoseKeypoints_y.max() - self.TPoseKeypoints_y.min()) / self.HEIGHT) < 0.1):
            print("detected tpose")
            return True
        else:
            return False

    def isFieldGoal(self):
        elbow2elbow_indexes = [body25[x] for x in ['Neck','RShoulder','RElbow','LElbow','LShoulder']]
        a = self.keypoints[0,elbow2elbow_indexes,1].flat
        self.elbowToElbow_y = a[a>0]
        
        
        isElbowsCorrect = False
        if (self.elbowToElbow_y >= 4) and (((self.elbowToElbow_y.max() - self.elbowToElbow_y.min()) / self.HEIGHT) < 0.07):
            isElbowsCorrect = True

        threshhold = 0.07
        isHandsCorrect = False
        if (self.keypoints[0,body25['RWrist'],0]-self.keypoints[0,body25['RElbow'],0]) < threshhold:
            if (self.keypoints[0,body25['LWrist'],0]-self.keypoints[0,body25['LElbow'],0]) < threshhold:
                if (self.keypoints[0,body25['RWrist'],1] < self.keypoints[0,body25['Nose'],1]):
                    if (self.keypoints[0,body25['LWrist'],1] < self.keypoints[0,body25['Nose'],1]):
                        isHandsCorrect = True

        return (isElbowsCorrect and isHandsCorrect)

cv2.namedWindow("preview")
cap = cv2.VideoCapture(0)

if cap.isOpened():
	rval, img = cap.read()
else:
	rval = False

WIDTH = cap.get(3)
HEIGHT = cap.get(4)
sep = WIDTH/10

gestured = "none"

if DEBUG_MAIN:
    print("into loop:")
    
print WIDTH,HEIGHT

if DEBUG_MQTT:
    client.publish(return_topic, payload= ('HELLLOO'), qos=0, retain=False)

#while rval and not waiting_for_target:
while rval:
    # Read new image
    rval, img = cap.read()
    # Output keypoints and the image with the human skeleton blended on it
    keypoints, output_image = openpose.forward(img, True)

    # Display the image
    cv2.imshow("preview", output_image)

    if DEBUG_MAIN:
        print(keypoints.size)

    #check for gesture
    # if keypoints.size > 0:
    #     gesture = ProcessKeyPoints(keypoints, WIDTH, HEIGHT)
    #     if (gesture.isTPose()):
    #         print("TPOSE DETECTED...")
    #         gestured = "TPose"
    #         waiting_for_target = True

    # with more gestures, %target_gesture from MQTT Unity
    # waiting_for_target = False
    # target_gesture = "tpose" 
    if not waiting_for_target and keypoints.size > 0:
        gesture = ProcessKeyPoints(keypoints, WIDTH, HEIGHT)
        # print("checking for: "+target_gesture)
        if ( gesture.checkFor(target_gesture)):
            # send gesture correct to unity
            client.publish(return_topic, payload= ('correct'), qos=0, retain=False)
            # print("Correct")
            waiting_for_target = True

    # if keypoints.size > 0:
    #     nose_x = keypoints[0][0][0]
    #     region = 10 - int(nose_x/sep)
    # #print(region)
    # client.publish('localization',  payload= (region), qos=0, retain=False)

    # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the keypoints of all the people on that image
    # print(keypoints)
    # print()
    # print(keypoints.shape) # (1L, 25L, 3L)

    key = cv2.waitKey(20)
    if key == 27:
    	break

if DEBUG_MAIN:
    print(gestured)

cap.release()
cv2.destroyWindow("preview")
