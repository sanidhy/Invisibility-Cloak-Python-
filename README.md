# Invisibility-Cloak-Python-
Invisibility Cloak(Python)





# Prerequisites for the cloth:
Choose a cloth of one color only and suppose the color of the cloth is red then ensure that your background does not contain any red color. Because if the background contains that color then it will cause problems.
In this project, we are using red color cloth but you can make any color we just have to change the values for the visibilities of the color and it can be changed easily.
Now After choosing the cloth we need to select IDE for this Project and install some of the Libraries to make this work.

There are 3 things which we need on our system 

1.Python version 3.0.0 or above
2.OpenCV(To install this we need to run a command which is discussed in further part)
3.Numpy(to handle all the operations)

# Step 1: Importing the Libraries
# Import Libraries
import numpy as np
import cv2
import time

# Step 2: Using the WebCam to take the Video Feed
cap = cv2.VideoCapture(0)
time.sleep(2)     
background = 0

# Step 3: Capturing the Background
for i in range(50):
    ret, background = cap.read()
    
# Step 4: Capturing the video feed using Webcam

while(cap.isOpened()): 
    ret, img = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
Step 5: Setting the values for the cloak and making masks
#all this Comes in the while loop
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255]) # values is for red colour Cloth
    mask1 = cv2.inRange(hsv, lower_red,upper_red)
    lower_red = np.array([170,120,70])
    upper_red =  np.array([180,255,255])
mask2 = cv2.inRange(hsv,lower_red,upper_red)
#Combining the masks so that It can be viewd as in one frame
    mask1 = mask1 +mask2
#After combining the mask we are storing the value in deafult mask.


# Step 6: Using Morphological Transformations to remove noise from the cloth and unnecessary Details.
mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8), iterations = 2)
mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE,np.ones((3,3),np.uint8), iterations = 1)

mask2 =cv2.bitwise_not(mask1)

# Step 7: Combining the masks and showing them in one frame
res1 = cv2.bitwise_and(background,background,mask=mask1)
res2 = cv2.bitwise_and(img,img,mask=mask2)
final_output = cv2.addWeighted(res1,1,res2,1,0)
cv2.imshow('Invisible Cloak',final_output)
k = cv2.waitKey(10)
if k==27:
    break
cap.release()
cv2.destroyAllWindows()


