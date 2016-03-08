from Servo import *
from User import *
import cv2
import time
import mraa
import os
import sys
import numpy as np

BUTTON_GPIO1 = 13
btn = mraa.Gpio(BUTTON_GPIO1)
btn.dir(mraa.DIR_IN)

LED_GPIO1 = 12
led1 = mraa.Gpio(LED_GPIO1)
led1.dir(mraa.DIR_OUT)

LED_GPIO2 = 8
led2 = mraa.Gpio(LED_GPIO2)
led2.dir(mraa.DIR_OUT)

LED_GPIO3 = 7
led3 = mraa.Gpio(LED_GPIO2)
led3.dir(mraa.DIR_OUT)

LED_GPIO2 = 6
led4 = mraa.Gpio(LED_GPIO2)
led4.dir(mraa.DIR_OUT)

ledState1 = False
led1.write(0)

ledState2 = False
led2.write(0)

ledState3 = False
led3.write(0)

ledState4 = False
led4.write(0)

def getButtonPress():
    
    while 1:
        
        if (btn.read() != 0):
            continue
        else:
            time.sleep(0.05)
            if (btn.read() == 1):
                return
            else:
                continue

def normalize(X, low, high, dtype=None):
    """Normalizes a given array in X to a value between low and high."""
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)		# normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))			# scale to [low...high].
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)

def blink(blinkLed):
	blinkLed.dir(mraa.DIR_OUT)     # Set the direction as output
	ledState = False               # LED is off to begin with
	blinkLed.write(0)
	# One infinite loop coming up
	while True:
		if ledState == False:
			# LED is off, turn it on
			blinkLed.write(1)
			ledState = True        # LED is on
		else:
			blinkLed.write(0)
			ledState = False
		# Wait for some time 
		time.sleep(1)

def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.
    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes
    Returns:
        A list [X,y]
            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    for filename in os.listdir(path):
		try:
			if (filename != 'face.png'):
				print filename
				face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
				im = cv2.imread(os.path.join(path, filename))
				cv_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
				sim = cv_im
				faces = face_cascade.detectMultiScale(cv_im,scaleFactor=1.3,minNeighbors=5)
				for (p,q,r,s) in faces:
					 cv2.rectangle(cv_im,(p,q),(p+r,q+s),(150,125,0),2)#drawing a rectangle indicating face
					 sim = cv_im[q:q+s, p:p+r]
				# resize to given size (if given)
				if (sz is not None):
					im = cv2.resize(sim, sz)
				X.append(np.frombuffer(im, dtype=np.uint8))
				y.append(c)
		except IOError, (errno, strerror):
			print "I/O error({0}): {1}".format(errno, strerror)
		except:
			print "Unexpected error:", sys.exc_info()[0]
			raise
		c = c+1
		led2.write(0)
		ledState2 = False
    return [X,y]

def servoturn():
	servo = Servo("s1")
	servo.attach(3)
	time.sleep(0.05)
		
        # From 0 to 180 degrees
	for angle in range(0,180):
		servo.write(angle)
		time.sleep(0.05)
		
def facial_rec():
	out_dir = None
	servo = Servo("s1")
	servo.attach(3) #attaches to pin 3 in pwm
	print("New picture has been found in library")
	# if len(sys.argv) < 2:
		# print "USAGE: facerec_demo.py </path/to/images> [</path/to/store/images/at>]"
		# sys.exit()
		# Now read in the image data. This must be a valid path!
	[X,y] = read_images(sys.argv[1], (200, 100))

	face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

	img = cv2.imread(os.path.join(sys.argv[1], 'face.png'))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (p,q,r,s) in faces:
		cv2.rectangle(gray,(p,q),(p+r,q+s),(150,125,0),2)#drawing a rectangle indicating face
		sample = gray[q:q+s, p:p+r]
	# resize to given size (if given)
	sample = cv2.resize(gray, (200,100))

	sIm = np.frombuffer(sample, dtype=np.uint8)
	y = np.asarray(y, dtype=np.int32)
	if len(sys.argv) == 3:
		out_dir = sys.argv[2]
	model = cv2.createFisherFaceRecognizer()
	model.train(np.asarray(X), np.asarray(y))

	[p_label, p_confidence] = model.predict(np.asarray(sIm)) #replace sIm with another 
	print "Predicted label = %d (confidence=%.2f)" % (p_label, p_confidence)

	if(p_confidence > 0):
		print "Face does not match"
		blinkLed = mraa.Gpio(LED_GPIO3) # Get the LED pin object
		blink(blinkLed)
	else: 
		print "Face Matches!!!!!"
		
		servoturn()
		
	if out_dir is None:
		cv2.waitKey(0)
		
if __name__ == '__main__':
	ans = True
	while ans:
		print('''
		1 Take picture to process recognition
		2 Add new user 
		''')
		ans = input()
		
		if ans == 1: 
			print("Press the button to take picture")
			getButtonPress()
			if btn.read() == 1:
				led1.write(1)			#turn on led when button is pressed
				ledState1 = True
				print("Picture has been taken. Please wait for facial recognition")
				cap = cv2.VideoCapture(0)
				ret, frame = cap.read()
				cv2.imwrite('/home/root/FacialRecognition/lib/face.png',frame)
				cap.release()
				led1.write(0)			#turn off led when the camera is captured	
				led2.write(1)				#turn on led when processing
				ledState2 = True
				facial_rec()
			else:
				led1.write(0)
				ledState1 = False
				print("You didn't press the button correctly. Please try again")
				
			time.sleep(0.005)
		elif ans == 2:
			print("\nAdd user")
		
		elif ans != "":
			print("\nNot a valid choice")