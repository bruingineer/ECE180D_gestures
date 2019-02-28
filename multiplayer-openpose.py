# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import argparse

from time import time

import numpy as np 
import paho.mqtt.client as mqtt

# contants to enable debug options
# TODO: add these to the arguments
DEBUG_MQTT = False
DEBUG_MAIN = False
DEBUG_PROCESS_KEYPOINTS = False
MQTT_ENABLE = True

waiting_for_target = True

# mqtt setup
if MQTT_ENABLE:
    ip = "131.179.29.167"
    port = 1883

    # mqtt topics for gestures
    target_topic = 'gesture'
    return_topic = 'gesture_correct'
    target_gesture = "stop"

# on connect callback for mqtt
def on_connect(client, userdata, flags, rc):
    print("Connected with rc: {}".format(str(rc)))
    print("Connection returned result: {}".format(connack_string(rc)))
    client.isConnected = True
    if DEBUG_MQTT:
        print("DEBUG_MQTT * on_connect: target_gesture: {}".format(target_gesture))

# on message callback for mqtt
def on_message(client, userdata, msg):
    print("msp received: "+msg.topic+" "+str(msg.payload))

    # if the message is on the gesture topic, process it
    if msg.topic == target_topic:
        #print("HELLOO")
        global target_gesture
        global waiting_for_target
        #global DEBUG_MQTT
        target_gesture = str(msg.payload)
        
        if DEBUG_MQTT:
            print("DEBUG_MQTT * on_message: message from {}\ntarget_gesture={}".format(msg.topic, target_gesture))

        print(target_gesture)
        if target_gesture == "stop":
            waiting_for_target = True
        else:
            waiting_for_target = False
            if DEBUG_MQTT:
                print("set waiting to "+waiting_for_target+". gesture = "+target_gesture)

# connect to mqtt server
def connect_to_server(_ip, _port):
    client = mqtt.Client(client_id = 'openpose')
    client.on_connect = on_connect
    client.on_message = on_message
    print("connect_to_server: target_gesture = {}".format(target_gesture))
    client.connect(_ip, _port, 60)
    client.subscribe(target_topic, qos=0)
    return client

# dictionary to convert body part names to the body_25 indexes
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
    """
    reference for coordinates given in keypoints np array
    (WIDTH,0)---------(0,0)
          |             |
          |             |
    (WIDTH,HEIGHT)----(0,HEIGHT)

    keypoints shape [people x parts x index] == (1L, 25L, 3L)
    index[] = [x , y , confidence]
    """

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
        #print ("CheckFor " + _target_gesture)
        _target_gesture = _target_gesture.lower()
        if _target_gesture == "tpose":
            return self.isTPose()
        elif _target_gesture == "fieldgoal":
            return self.isFieldGoal()
        elif _target_gesture == "righthandwave":
            return self.isRightHandRightToLeftWave()
        elif _target_gesture == "lefthandraise":
            return self.isRaiseLeftHand()
        elif _target_gesture == "righthandraise":
            return self.isRaiseRightHand()
        else:
            print("gesture "+_target_gesture+" not recognized.")

    def isRightHandRightToLeftWave(self):
        x = []
        y = []
        if DEBUG_PROCESS_KEYPOINTS:
            print("Wave - last3frames length: {}".format(len(self.last_3_frames)))
        
        # get the right wrist x-y coordinates from the last 3 frames array
        for i in range(len(self.last_3_frames)):
            frame = self.last_3_frames[i][1]
            x.append(frame[0,body25['RWrist'],0])
            y.append(frame[0,body25['RWrist'],1])

        if DEBUG_PROCESS_KEYPOINTS:
            print("Wave - processed x: {0}".format(x))
        
        npx = np.array([z for z in x if z>0])
        npy = np.array([z for z in y if z>0])

        # TODO: normalize threshold to skeleton size
        if len(npy) > 1:
            # check for minimal y movement
            if ( ((npy.max() - npy.min())/self.HEIGHT) < 0.3 ):
                # check for x coordinates are in order which means movement in one directin
                # check for movement across .4 of the screen width
                # TODO normalize sizes
                if ( np.array_equal(np.sort(npx, axis=None)[::-1], npx) ) and ( ((npx.max() - npx.min())/self.WIDTH) > 0.4 ):
                    return True
        return False


    # checks for arms straight out from sides, using passed in keypoints to class
    # returns true if in TPose
    def isTPose(self):
        #print ("checking for tpose")
        # 1D array: y of [body25 1 thru 7]
        parts = [body25[x] for x in ['RWrist','RElbow','RShoulder','Neck','LShoulder','LElbow','LWrist']]

        # get y
        a = self.keypoints[0, parts, 1].flat
        TPoseKeypoints_y = a[a > 0]
        # get x
        a = self.keypoints[0, parts, 0].flat
        TPoseKeypoints_x = a[a > 0]
        
        if DEBUG_PROCESS_KEYPOINTS:
            print('tpose - y: {}'.format(self.TPoseKeypoints_y))
            print('tpose - x: {}'.format(self.TPoseKeypoints_y1))

        # check for at least 6 of the 8 points are detected and their y variation is within threshold
        threshold = 0.1
        if (TPoseKeypoints_y.size >= 6) and (abs(((TPoseKeypoints_y.max() - TPoseKeypoints_y.min()) / self.HEIGHT)) < threshold):
            if DEBUG_PROCESS_KEYPOINTS:
                print("tpose - testing for x order")
            if (np.array_equal(np.sort(TPoseKeypoints_x, axis=None), TPoseKeypoints_x)):
                return True
        else:
            return False

    def isFieldGoal(self):
        # get parts needed for this pose
        elbow2elbow_indexes = [body25[x] for x in ['Neck','RShoulder','RElbow','LElbow','LShoulder']]
        a = self.keypoints[0,elbow2elbow_indexes,1].flat
        self.elbowToElbow_y = a[a>0]
        
        # check that elbows and shoulders are horizontal within threshold
        threshold = 0.2
        isElbowsCorrect = False
        if (self.elbowToElbow_y.size >= 4) and (abs(((self.elbowToElbow_y.max() - self.elbowToElbow_y.min()) / self.HEIGHT)) < threshold):
            isElbowsCorrect = True

        isHandsCorrect = False
        # check if wrist and elbox x coords are within in threshold
        if np.count_nonzero(self.keypoints[0,[body25[x] for x in ['RElbow','RWrist','LWrist','LElbow']],0]) == 4:
            if DEBUG_PROCESS_KEYPOINTS:
                print('field goal - non zero check')
            if (abs(self.keypoints[0,body25['RWrist'],0]-self.keypoints[0,body25['RElbow'],0]) / self.WIDTH) < threshold:
                if DEBUG_PROCESS_KEYPOINTS:
                    print("field goal - Right pass")
                if (abs(self.keypoints[0,body25['LWrist'],0]-self.keypoints[0,body25['LElbow'],0]) / self.WIDTH) < threshold:
                    if DEBUG_PROCESS_KEYPOINTS:
                        print('field goal - left pass')
                    # check if wrists are above nose
                    if (self.keypoints[0,body25['RWrist'],1] < self.keypoints[0,body25['Nose'],1]):
                        if (self.keypoints[0,body25['LWrist'],1] < self.keypoints[0,body25['Nose'],1]):
                            isHandsCorrect = True

        return isHandsCorrect and isElbowsCorrect

    #TODO normalize and add checks for other arm is down
    # left hand raised
    def isRaiseLeftHand(self):
        indexes = [body25[x] for x in ['LElbow','LShoulder','LWrist']]
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

    # right hand raised
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

def main():

    global waiting_for_target
    global target_gesture

    # numpy suppress sci notation, set 1 decimal place
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=1)

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--op_dir", default='op_cuda_jose', help="Path to compiled OpenPose library \
        folder which includes the lib, x64/Release, and python folders.")
    parser.add_argument("--gesture", default=None, help="Target Gesture to search for during testing.")
    parser.add_argument("--localization", default=True, help="If True, this enables the localization functionality.")
    parser.add_argument("--ip", default=None, help="Set the ip of the Mqtt server.")

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

    # connect mqtt server
    client = connect_to_server(args[0].ip, port)
    client.loop_start()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    cv2.namedWindow("Synchro - Player 0")
    # cv2.namedWindow("Synchro - Player 1")
    cap0 = cv2.VideoCapture(0)
    # cap1 = cv2.VideoCapture(1)

    if cap0.isOpened():
        rval0, img0 = cap0.read()
        print("cap0 open")
    else:
    	rval0 = False

    # if cap1.isOpened():
    #     rval1, img1 = cap1.read()
    #     print("cap1 open")
    # else:
    #     rval1 = False

    WIDTH0 = cap0.get(3)
    HEIGHT0 = cap0.get(4)

    # WIDTH1 = cap1.get(3)
    # HEIGHT1 = cap1.get(4)
    # sep1 = WIDTH1/10

    if DEBUG_MAIN:
        print("into loop:")
        
    print("0 w x h: {0} x {1}".format(WIDTH0,HEIGHT0))
    # print("1 w x h: {0} x {1}".format(WIDTH1,HEIGHT1))

    if MQTT_ENABLE and DEBUG_MQTT:
        client.publish(return_topic, payload= ('HELLLOO'), qos=0, retain=False)

    gesture0 = keypointFrames()
    # gesture1 = keypointFrames()

    #while rval and not waiting_for_target:
    while rval0:
        # Read new image
        rval0, img0 = cap0.read()
        # rval1, img1 = cap1.read()

        # Process Image
        datum0 = op.Datum()
        datum0.cvInputData = img0
        opWrapper.emplaceAndPop([datum0])        

        # datum1 = op.Datum()
        # datum1.cvInputData = img1
        # opWrapper.emplaceAndPop([datum1])

        # get keypoints and the image with the human skeleton blended on it
        main_keypoints = datum0.poseKeypoints
        # keypoints_1 = datum1.poseKeypoints

        # Display the image
        flipped0 = cv2.flip(datum0.cvOutputData, 1)
        cv2.imshow("Synchro - Player 0", flipped0)
        # flipped_1 = cv2.flip(datum1.cvOutputData, 1)
        # cv2.imshow("Synchro - Player 1", flipped_1)

        # check for gesture
        # with more gestures, _target_gesture from MQTT Unity
        if args[0].gesture is not None:
            waiting_for_target = False
            target_gesture = args[0].gesture 

        #print ("waiting for target " + str(waiting_for_target))
        #print(main_keypoints.size)
        if not waiting_for_target and main_keypoints.size > 1:
            #print("about to call checkFor")
            gesture0.add(main_keypoints, WIDTH0, HEIGHT0)
            if DEBUG_MAIN:
                print("main - checking for: {}".format(target_gesture))
            if ( gesture0.checkFor(target_gesture) ):
                # send gesture correct to unity
                if MQTT_ENABLE:
                    client.publish(return_topic, payload= ('correct'), qos=0, retain=False)
                print("{target_gesture}: Correct".format(target_gesture=target_gesture))
                waiting_for_target = True

        # if using this script for localization
        if args[0].localization:
            if main_keypoints.size > 1:
                nose_x = main_keypoints[0][0][0]
                region = 10 - int(10*nose_x/WIDTH0)
            # print(region)
            if MQTT_ENABLE:
                client.publish('localization',  payload= (region), qos=0, retain=False)
                print (region)

        # break loop and exit when ESC key is pressed
        key = cv2.waitKey(20)
        if key == 27:
        	break

    cap0.release()
    # cap1.release()
    # cv2.destroyWindow("Synchro - Player 1")
    cv2.destroyWindow("Synchro - Player 0")

if __name__ == '__main__':
    main()