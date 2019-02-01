# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
from collections import deque
from time import time
from threading import Thread

import numpy as np 
import paho.mqtt.client as mqtt

DEBUG_MQTT = False
DEBUG_MAIN = False
DEBUG_PROCESS_KEYPOINTS = False
MQTT_ENABLE = False

def on_connect(client, userdata, flags, rc):
    print("Connected with rc: "+str(rc))
    print("Connection returned result: {}".format(connack_string(rc)))
    client.isConnected = True
    if DEBUG_MQTT:
        print("DEBUG_MQTT * on_connect: target_gesture: "+target_gesture)


def on_message(client, userdata, msg):
    print("msp received: "+msg.topic+" "+str(msg.payload))
    if msg.topic == target_topic:
        global target_gesture
        global waiting_for_target
        target_gesture = str(msg.payload)
        
        if DEBUG_MQTT:
            print("DEBUG_MQTT * on_message: message from "+msp.topic+"\ntarget_gesture="+target_gesture)

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
    # class _frame:
    #     def __init__(self, kps):
    #         self.keypoints = kps
    #         self.timestamp = time() 

    def __init__(self):
        self.last_3_frames = []

    def add(self, _keypoints, inputWIDTH, inputHEIGHT):
        self.keypoints = np.array(_keypoints)
        # print(self.keypoints)
        # print('*********')
        if len(self.last_3_frames) < 3:
            self.last_3_frames.append((time(),self.keypoints))
            #print(self.last_3_frames)
        else:
            self.last_3_frames.pop(0)
            self.last_3_frames.append((time(),self.keypoints))

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
        elif _target_gesture == "leftHandRaise":
            return self.isRaiseLeftHand()
        elif _target_gesture == "rightHandRaise":
            return self.isRaiseRightHand()
        else:
            print("gesture "+_target_gesture+" not recognized.")

    def isRightHandRightToLeftWave(self):
        x = []
        y = []
        print("{0} length".format(len(self.last_3_frames)))
        for i in range(len(self.last_3_frames)):
            frame = self.last_3_frames[i][1]
            x.append(frame[0,body25['RWrist'],0])
            y.append(frame[0,body25['RWrist'],1])
        print("x: {0}".format(x))
        npx = np.array([z for z in x if z>0])
        npy = np.array([z for z in y if z>0])
        # print(npy)
        # print(npx)
        if len(npy) > 1:
            if ( ((npy.max() - npy.min())/self.HEIGHT) < 0.3 ):
                if ( np.array_equal(np.sort(npx, axis=None)[::-1], npx) ) and ( ((npx.max() - npx.min())/self.WIDTH) > 0.4 ):
                    return True
        return False


    # checks for arms straight out from sides, using passed in keypoints to class
    # returns true if in TPose
    def isTPose(self):
        # 1D array: y of [body25 1 thru 7]
        # self.TPoseKeypoints_y = self.keypoints[0,1:8,1][ self.keypoints[0,1:8,1].flat > 0 ]
        # OR
        parts = [body25[x] for x in ['RWrist','RElbow','RShoulder','Neck','LShoulder','LElbow','LWrist']]
        a = self.keypoints[0, parts, 1].flat
        TPoseKeypoints_y = a[a > 0]
        a = self.keypoints[0, parts, 0].flat
        TPoseKeypoints_x = a[a > 0]
        # print(self.TPoseKeypoints_y)
        # print(self.TPoseKeypoints_y1)
        # print("")
        threshold = 0.1
        if (TPoseKeypoints_y.size >= 6) and (abs(((TPoseKeypoints_y.max() - TPoseKeypoints_y.min()) / self.HEIGHT)) < threshold):
            # print("detected tpose")
            print("testing for x order")
            if (np.array_equal(np.sort(TPoseKeypoints_x, axis=None), TPoseKeypoints_x)):
                return True
        else:
            return False

    def isFieldGoal(self):
        elbow2elbow_indexes = [body25[x] for x in ['Neck','RShoulder','RElbow','LElbow','LShoulder']]
        # print(self.keypoints.size)
        a = self.keypoints[0,elbow2elbow_indexes,1].flat
        self.elbowToElbow_y = a[a>0]
        
        threshold = 0.2
        isElbowsCorrect = False
        if (self.elbowToElbow_y.size >= 4) and (abs(((self.elbowToElbow_y.max() - self.elbowToElbow_y.min()) / self.HEIGHT)) < 0.2):
            isElbowsCorrect = True

        isHandsCorrect = False
        # check if hand and elbox x coords are within in threshold
        if np.count_nonzero(self.keypoints[0,[body25[x] for x in ['RElbow','RWrist','LWrist','LElbow']],0]) == 4:
            # print('non zero check')
            if (abs(self.keypoints[0,body25['RWrist'],0]-self.keypoints[0,body25['RElbow'],0]) / self.WIDTH) < threshold:
                # print("Right pass")
                if (abs(self.keypoints[0,body25['LWrist'],0]-self.keypoints[0,body25['LElbow'],0]) / self.WIDTH) < threshold:
                    # print('left pass')
                    # check if wrists are above nose
                    if (self.keypoints[0,body25['RWrist'],1] < self.keypoints[0,body25['Nose'],1]):
                        if (self.keypoints[0,body25['LWrist'],1] < self.keypoints[0,body25['Nose'],1]):
                            isHandsCorrect = True

    def isRaiseLeftHand(self):
        indexes = [body25[x] for x in ['LElbow','LShoulder','LWrist']]
        # print(self.keypoints.size)
        a = self.keypoints[0,indexes,1].flat
        y = a[a>0]
        a = self.keypoints[0,indexes,0].flat
        x = a[a>0]
        
        #checking for low change in x ==> arm is straight up
        threshold = 0.2
        if ( y.size == 3 ) and (abs(((x.max() - x.min()) / self.HEIGHT)) < threshold):
            # check if wrists are above nose
            if (self.keypoints[0,body25['LWrist'],1] < self.keypoints[0,body25['Nose'],1]):
                return True

        return False

    def isRaiseRightHand(self):
        indexes = [body25[x] for x in ['RElbow','RShoulder','RWrist']]
        # print(self.keypoints.size)
        a = self.keypoints[0,indexes,1].flat
        y = a[a>0]
        a = self.keypoints[0,indexes,0].flat
        x = a[a>0]
        
        #checking for low change in x ==> arm is straight up
        threshold = 0.2
        if ( y.size == 3 ) and (abs(((x.max() - x.min()) / self.HEIGHT)) < threshold):
            # check if wrists are above nose
            if (self.keypoints[0,body25['RWrist'],1] < self.keypoints[0,body25['Nose'],1]):
                return True

        return False

"""
BEGIN THREAD
"""
def op_thread(_opWrapper, op, tid):
    cv2.namedWindow("Synchro - Player {}".format(tid))
    # cv2.namedWindow("Synchro - Player 1")
    cap0 = cv2.VideoCapture(tid)
    # cap1 = cv2.VideoCapture(1)

    if cap0.isOpened():
        rval0, img0 = cap0.read()
        print("cap{} open".format(tid))
    else:
        rval0 = False

    # if cap1.isOpened():
    #     rval1, img1 = cap1.read()
    #     print("cap1 open")
    # else:
    #     rval1 = False



    WIDTH0 = cap0.get(3)
    HEIGHT0 = cap0.get(4)
    sep0 = WIDTH0/10

    # WIDTH1 = cap1.get(3)
    # HEIGHT1 = cap1.get(4)
    # sep1 = WIDTH1/10

    if DEBUG_MAIN:
        print("into loop:")
        
    print("0 w x h: {0} x {1}".format(WIDTH0,HEIGHT0))
    # print("1 w x h: {0} x {1}".format(WIDTH1,HEIGHT1))

    if MQTT_ENABLE and DEBUG_MQTT:
        client.publish(return_topic, payload= ('HELLLOO'), qos=0, retain=False)

    # gesture0 = keypointFrames()

    #while rval and not waiting_for_target:
    while rval0:
        # Read new image
        rval0, img0 = cap0.read()

        # Process Image
        datum0 = op.Datum()
        datum0.cvInputData = img0
        _opWrapper.emplaceAndPop([datum0])        

        # get keypoints and the image with the human skeleton blended on it
        main_keypoints = datum0.poseKeypoints

        # Display the image
        flipped0 = cv2.flip(datum0.cvOutputData, 1)
        cv2.imshow("Synchro - Player {}".format(tid), flipped0)

        # break loop and exit when ESC key is pressed
        key = cv2.waitKey(20)
        if key == 27:
            break

    cap0.release()
    cv2.destroyWindow("Synchro - Player {}".format(tid))
"""
END THREAD
"""
def main():

    # numpy suppress sci notation, set 1 decimal place
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=1)

    waiting_for_target = True

    # mqtt setup
    if MQTT_ENABLE:
        ip = "131.179.28.219"
        port = 1883

        target_topic = 'gesture'
        return_topic = 'gesture_correct'
        target_gesture = "stop"

        client = connect_to_server(ip, port)
        client.loop_start()

    # Flags
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--op_dir", default='op_cuda_jose', help="Path to compiled OpenPose library \
        folder which includes the lib, x64/Release, and python folders.")
    parser.add_argument("--gesture", default=None, help="Target Gesture to search for during testing.")
    parser.add_argument("--localization", default=False, help="Add argument to use this script for localization.")
    args = parser.parse_known_args()

    # Remember to add your installation path here
    # Adds directory of THIS script to OS PATH (to search for necessary DLLs & models)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(1, dir_path + "\\" + args[0].op_dir +"\\python\\openpose\\Release")
    os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/' + args[0].op_dir + '/x64/Release;' +  dir_path + '/' + args[0].op_dir +'/bin;'

    try:
        import pyopenpose as op 
    except:
        raise Exception('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')


    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = dir_path + "/models/"
    #params["frame_flip"] = "True"
    params["model_pose"] = "BODY_25"
    params["net_resolution"] = "-1x160"
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

    thread_0 = Thread(target=op_thread, args = (opWrapper, op, 0, ))
    thread_1 = Thread(target=op_thread, args = (opWrapper, op, 1, ))
    thread_0.start()
    thread_1.start()
    thread_0.join()
    thread_1.join()

    exit()

if __name__ == '__main__':
    main()