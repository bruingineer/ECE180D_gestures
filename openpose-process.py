# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
from collections import deque

import numpy as np 
import paho.mqtt.client as mqtt

# Remember to add your installation path here
# Adds directory of THIS script to OS PATH (to search for necessary DLLs & models)
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, dir_path + "\\op_cpu_only\\python\\openpose\\Release")
os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/op_cpu_only/x64/Release;' +  dir_path + '/op_cpu_only/bin;'

try:
    import pyopenpose as op 
except:
    raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')



def on_connect(client, userdata, flags, rc):
    print("Connected with rc: "+str(rc))
    global CONNECTED
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
#    global client
    client = mqtt.Client(client_id = 'openpose')
    client.on_connect = on_connect
    client.on_message = on_message
    print("connect_to_server: target_gesture = "+target_gesture)
    client.connect(ip, port, 60)
    client.subscribe(target_topic, qos=0)
    return client
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
class keypointFrames:
    # reference for coordinates given in keypoints np array
    # (0,0)-------->(WIDTH,0)
    #   |               |
    #   |               |
    # (0,HEIGHT)--(WIDTH,HEIGHT)

    # keypoints shape [people x parts x index] == (1L, 25L, 3L)
    # index[] = [x , y , confidence]
    # keypoints for BODY_25
    
    def __init__(self):
        self.last_3_frames = []

    def add(self, _keypoints, inputWIDTH, inputHEIGHT):
        self.keypoints = np.array(_keypoints)
        # print(self.keypoints)
        # print('*********')
        if len(self.last_3_frames) < 3:
            self.last_3_frames.append(self.keypoints)
            #print(self.last_3_frames)
        else:
            self.last_3_frames.pop(0)
            self.last_3_frames.append(self.keypoints)

        #np.array(keypoints)
        # self.num_people = self.keypoints.shape[1]
        self.WIDTH = inputWIDTH
        self.HEIGHT = inputHEIGHT

    def checkFor(self, _target_gesture):
        if _target_gesture == "tpose":
            return self.isTPose()
        elif _target_gesture == "fieldgoal":
            return self.isFieldGoal()
        elif _target_gesture == "rightHandWave":
            return self.isRightHandRightToLeftWave()
        else:
            print("gesture "+_target_gesture+" not recognized.")

    def isRightHandRightToLeftWave(self):
        x = []
        y = []
        for i in range(len(self.last_3_frames)-1):
            frame = self.last_3_frames[i]
            x.append(frame[0,[body25['RWrist']],0])
            y.append(frame[0,body25['RWrist'],1])
        npx = np.array([z for z in x if z>0])
        npy = np.array([z for z in y if z>0])

        if len(npy) > 1:
            if ( ((npy.max() - npy.min())/self.HEIGHT) < 0.1 ):
                if ( np.array_equal(np.sort(npx, axis=None), npx) ) and ( ((npx.max() - npx.min())/self.WIDTH) > 0.6 ):
                    return True
        return False


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
        threshold = 0.1
        if (self.TPoseKeypoints_y.size >= 6) and (abs(((self.TPoseKeypoints_y.max() - self.TPoseKeypoints_y.min()) / self.HEIGHT)) < threshold):
            print("detected tpose")
            return True
        else:
            return False

    def isFieldGoal(self):
        elbow2elbow_indexes = [body25[x] for x in ['Neck','RShoulder','RElbow','LElbow','LShoulder']]
        print(self.keypoints.size)
        a = self.keypoints[0,elbow2elbow_indexes,1].flat
        self.elbowToElbow_y = a[a>0]
        
        threshold = 0.2
        isElbowsCorrect = False
        if (self.elbowToElbow_y.size >= 4) and (abs(((self.elbowToElbow_y.max() - self.elbowToElbow_y.min()) / self.HEIGHT)) < 0.2):
            isElbowsCorrect = True

        isHandsCorrect = False
        # check if hand and elbox x coords are within in threshold
        if np.count_nonzero(self.keypoints[0,[body25[x] for x in ['RElbow','RWrist','LWrist','LElbow']],0]) == 4:
            print('non zero check')
            if (abs(self.keypoints[0,body25['RWrist'],0]-self.keypoints[0,body25['RElbow'],0]) / self.WIDTH) < threshold:
                print("Right pass")
                if (abs(self.keypoints[0,body25['LWrist'],0]-self.keypoints[0,body25['LElbow'],0]) / self.WIDTH) < threshold:
                    print('left pass')
                    # check if wrists are above nose
                    if (self.keypoints[0,body25['RWrist'],1] < self.keypoints[0,body25['Nose'],1]):
                        if (self.keypoints[0,body25['LWrist'],1] < self.keypoints[0,body25['Nose'],1]):
                            isHandsCorrect = True

        return (isElbowsCorrect and isHandsCorrect)


def main():
    DEBUG_MQTT = False
    DEBUG_MAIN = False
    DEBUG_PROCESS_KEYPOINTS = False
    MQTT_ENABLE = False

    # numpy suppress sci notation, set 1 decimal place
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=1)

    waiting_for_target = True

    # mqtt setup
    if MQTT_ENABLE:
        ip = "131.179.28.219"
        port = 1883

        CONNECTED = False
        target_topic = 'gesture'
        return_topic = 'gesture_correct'
        target_gesture = "stop"

        client = connect_to_server(ip, port)
        client.loop_start()

    # Flags
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = dir_path + "/models/"
    params["frame_flip"] = "True"
    params["model_pose"] = "BODY_25"
    params["net_resolution"] = "-1x96"
    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    cv2.namedWindow("preview")
    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        rval, img = cap.read()
        print("cv open")
    else:
    	rval = False

    WIDTH = cap.get(3)
    HEIGHT = cap.get(4)
    sep = WIDTH/10

    gestured = "none"

    if DEBUG_MAIN:
        print("into loop:")
        
    print(WIDTH,HEIGHT)

    if MQTT_ENABLE and DEBUG_MQTT:
        client.publish(return_topic, payload= ('HELLLOO'), qos=0, retain=False)

    gesture = keypointFrames()
    #while rval and not waiting_for_target:
    while rval:
        # Read new image
        rval, img = cap.read()

        # Process Image
        datum = op.Datum()
        #imageToProcess = cv2.imread(args[0].image_path)
        #datum.cvInputData = imageToProcess
        flipped = cv2.flip(img,1)
        datum.cvInputData = flipped
        opWrapper.emplaceAndPop([datum])

        # Display Image
        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        # cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)
        # cv2.waitKey(0)

        # get keypoints and the image with the human skeleton blended on it
        main_keypoints = datum.poseKeypoints

        # Display the image
        cv2.imshow("preview", datum.cvOutputData)

        
        print(main_keypoints.size)
        print(main_keypoints)

        #check for gesture
        
        # with more gestures, %target_gesture from MQTT Unity
        waiting_for_target = False
        target_gesture = "fieldgoal" 
        if not waiting_for_target and main_keypoints.size > 1:
            gesture.add(main_keypoints, WIDTH, HEIGHT)
            # print("checking for: "+target_gesture)
            if ( gesture.checkFor(target_gesture)):
                # send gesture correct to unity
                if MQTT_ENABLE:
                    client.publish(return_topic, payload= ('correct'), qos=0, retain=False)
                print("{target_gesture}: Correct".format(target_gesture=target_gesture))
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

if __name__ == '__main__':
    main()