import ClassNeuralNetwork as CNN
import os
import sys
import torch
import cv2
import PIL
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def nothing(x): pass

def PreProcess(frame):
    frame = frame[ Y_start : Y_end, X_start : X_end] # Crop frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Change to greyscale

    _, frame_tresh = cv2.threshold(frame, 150, 255, cv2.THRESH_OTSU) # Apply treshholding
    frame_tresh = cv2.bitwise_not(frame_tresh)  # Invert
    
    contours, hierarchy = cv2.findContours(frame_tresh , mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE) # Find contours
    
    if len(contours) != 0:
        cv2.drawContours(frame, contours=contours, contourIdx=-1 , color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        c = max(contours, key = cv2.contourArea) # Get biggest contour
        rect = cv2.boundingRect(c) # Calculate bounding box
        r_size = int(1.25 * max([int(rect[2]),int(rect[3])])) # Get biggest dimension and multiply it by 1.5
        x = int(rect[0] - (r_size-rect[2])/2) # Calculate new offset values (x,y)
        if x < 0: x = 0
        y = int(rect[1] - (r_size-rect[3])/2)
        if y < 0: y = 0
    else: # if there is no contours
        x = 0
        y = 0
        r_size = max([frame.shape[0],frame.shape[1]])

    cv2.rectangle(frame,(x,y),(x+r_size,y+r_size),(0,255,0),2) # Draw rectangle (data from bounding box)
    cv2.imshow('Contours', frame)

    frame = frame_tresh[ y : y+r_size, x : x+r_size] # 2nd crop

    frame = cv2.resize(frame, img_size, interpolation = cv2.INTER_AREA) # Resize (Networks accept 28x28 images)
    # frame = (255 - frame) # Invert once more
    frame8bit = np.uint8(frame) # Change to 8-bit

    return frame8bit

def ApplyHUD(frame, frame_processed):
    # Draw "Crosshair"
    frame = cv2.line(frame, (0, int((frame.shape[0]-250)/2)), (frame.shape[1], int((frame.shape[0]-250)/2)), (0,0,255), 1)
    frame = cv2.line(frame, (int(frame.shape[1]/2), 0), (int(frame.shape[1]/2), frame.shape[0]-250), (0,0,255), 1)
    frame = cv2.circle(frame, (int(frame.shape[1]/2),int((frame.shape[0]-250)/2)), int((X_end - X_start)/2), (0,0,255), 2)

    # Draw viewportal rectange
    frame = cv2.rectangle(frame, (X_start,Y_start), (X_end,Y_end), (0,0,255), 1)

    # Draw Neural Network inpu image
    frame16bit = np.uint16(frame_processed) # Change to 16bit
    frame16bit = cv2.cvtColor(frame16bit, cv2.COLOR_GRAY2BGR) # Change to BGR

    frame16bit = cv2.resize(frame16bit, (250,250), interpolation = cv2.INTER_AREA) # Resize to 250x250
    frame[ frame.shape[0]-250:frame.shape[0], 0:250 ] = frame16bit # Draw resized input image
    frame = cv2.rectangle(frame, (0,frame.shape[0]-250), (250,frame.shape[0]), (0,0,255), 1)

    cv2.putText(frame, "Press ESC to quit", (260,frame.shape[0]-25), font, fontScale, fontColor, lineType)
    
    return frame

def PrepareData(frame_processed):
    # Prepare data for Neural Network
    TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize(img_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5])
    ])

    pilInput = PIL.Image.fromarray(frame_processed) # convert OpenCV image to PIL image
    NNinput = TRANSFORM(pilInput).unsqueeze(0) # convert PIL image to a PyTorch tensor
    data_loader = torch.utils.data.DataLoader(NNinput, batch_size = 1) # Data loader
    
    return data_loader

def OverlayResults(display_frame, result_str, all_result_str):

    # Draw text
    cv2.putText(display_frame, result_str, (X_start,Y_start+15), font, fontScaleSmall, fontColor, lineType)
    cv2.putText(display_frame, result_str, (260, display_frame.shape[0]-220), font, fontScale, fontColor, lineType)
    cv2.putText(display_frame, all_result_str, (260, display_frame.shape[0]-100), font, fontScaleSmall, fontColor, lineType)
    
    # Draw green rectangle on top of viewport rectangle
    display_frame = cv2.rectangle(display_frame, (X_start,Y_start), (X_end,Y_end), (0,255,0), 4)

    return display_frame

def NewWindowResults(display_frame, result_str):
    
    # Draw viewport in new window   
    new_img = display_frame[ Y_start : Y_end, X_start : X_end]
    cv2.imshow('Result', new_img)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", DEVICE)

#PATH = 'C:/Users/Robocode/Desktop/Sieci-Neuronowe'
PATH = sys.path[0]
filename = 'NN_MNIST.pth'
PATH = os.path.join(PATH, filename)

# Define network values
n_classes = 10 # Number of NN outputs
img_size = (28,28) # Size of NN input image
MIN_CONF_VALUE = 0.95 # Minimal value of confidence

viewport_size = (250,250) # Size of viewport

# Define text values
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontScaleSmall         = .33
fontColor              = (100,255,100)
lineType               = 2

#Create and load network
network = CNN.Network(n_classes).to(DEVICE) # move network to GPU if available
network.load_state_dict(torch.load(PATH))
#network.eval()

# define a video capture object
vid = cv2.VideoCapture(0)
if (vid.isOpened()== False):
  print("Error opening video stream or file")

#Define trackbars and options window
cv2.namedWindow('Options')
cv2.createTrackbar('Viewport size','Options',50,100,nothing)
cv2.createTrackbar('Minimal confidence','Options',99,100,nothing)

while(vid.isOpened()):
      
    # Capture the video frame
    ret, frame = vid.read()

    frame_disp = np.zeros((frame.shape[0]+250,frame.shape[1],frame.shape[2]), np.uint8)
    frame_disp[0:frame.shape[0], 0:frame.shape[1]] = frame

    #height, width, channels = frame.shape
    tmp_size = int(cv2.getTrackbarPos('Viewport size','Options') * 3.5 + 50)
    MIN_CONF_VALUE = int(cv2.getTrackbarPos('Minimal confidence','Options'))/100
    viewport_size = (tmp_size, tmp_size)

    #Calculate size for data crop
    X_start = int(frame.shape[1]/2 - viewport_size[1]/2)
    Y_start = int(frame.shape[0]/2 - viewport_size[0]/2)
    X_end = int(frame.shape[1]/2 + viewport_size[1]/2)
    Y_end = int(frame.shape[0]/2 + viewport_size[0]/2)

    key = cv2.waitKey(1)   
    if key == 27: break

    # PRE-PROCESSING    
    frame_processed = PreProcess(frame)
    
    # PREPARE DATA
    data_loader = PrepareData(frame_processed)

    # NEURAL NETWORK
    prediction = network(next(iter(data_loader)))
    #print(prediction)
    prediction_index = int(torch.argmax(prediction))
    prediction_value = float(torch.max(prediction))
    
    result_str = "R: " + str(prediction_index) + "; " + str(round(prediction_value*10000)/100) + "%"
    np_prediction = prediction.detach().numpy()[0]
    np_prediction = np.round(np_prediction * 10000)/100
    all_result_str = str(np_prediction)
    #print(all_result_str)
    #print(result_str)

    # APPLY HUD
    frame_disp = ApplyHUD(frame_disp, frame_processed)

    # DRAW RESULTS
    if prediction_value >= MIN_CONF_VALUE: 
        frame_disp = OverlayResults(frame_disp, result_str, all_result_str)
        NewWindowResults(frame_disp, result_str)
    cv2.imshow('Frame', frame_disp)
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()