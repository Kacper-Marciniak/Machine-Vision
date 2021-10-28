import os
import sys
import cv2
import numpy as np

import torch
import torchvision
import PIL

import ClassNeuralNetwork as CNN

SHOW_ALL = True

filename = 'NN_MNIST.pth'
PATH = os.path.join(sys.path[0], filename)

PATH_UI_NUMBERS = os.path.join(sys.path[0], 'CV_DATA/')

n_classes = 10 # Number of NN outputs
img_size = (28,28)

min_contour_area = 5**2
min_contour_size = 2
max_contour_size = 400
minimal_conf = .995
multiplier_contour_size = 1.0

result_buffer_size = 1

def nothing(x): pass

def LoadUINumbers(PATH_UI_NUMBERS):
    img_ui_list = list()
    for i in range(10):
        img = cv2.imread(PATH_UI_NUMBERS+str(i)+".png")
        img_ui_list.append(img)
    return img_ui_list


def DetectContours(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Change to greyscale
    image = cv2.GaussianBlur(image, (5, 5), 0) # Gauss blur
    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_OTSU) # Apply treshholding
    image = cv2.bitwise_not(image)  # Invert

    contours, hierarchy = cv2.findContours(image , mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE) # Find contours
    # contours, hierarchy = cv2.findContours(image , mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE) # Find contours

    new_contours = list()

    for c in contours:
        w = cv2.boundingRect(c)[2]
        h = cv2.boundingRect(c)[3]
        condition = bool(
            cv2.contourArea(c) > min_contour_area and w > min_contour_size and h > min_contour_size and w < max_contour_size and h < max_contour_size
            )
        if  condition:
            new_contours.append(c)

    if SHOW_ALL == True :
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = DrawContours(image, new_contours, color=(0,0,255), width=1)
        cv2.imshow('Tresh - Contours detected', image)

    return new_contours

def DrawContours(image, contours, color=(0,255,0), width=1):
    cv2.drawContours(image, contours, -1, (0,0,255), width)
    return image

def CreateRectangles(contours, multiplier_contour_size):
    rectangles = list()
    for c in contours:
        rect = cv2.boundingRect(c)
        r_size = int(multiplier_contour_size * max([int(rect[2]),int(rect[3])])) # Get biggest dimension and multiply it by 1.5
        x = int(rect[0] - (r_size-rect[2])/2) # Calculate new offset values (x,y)
        if x < 0: x = 0
        y = int(rect[1] - (r_size-rect[3])/2)
        if y < 0: y = 0
        rectangles.append((x,y,r_size,r_size))
    return rectangles
    

def DrawRectangles(image, rectangles, color=(0,255,0), width=1):
    for rect in rectangles:
        cv2.rectangle(image,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),color,width) # Draw rectangle (data from bounding box)

    return image

def CropROI(image, rect):    
    return image[ rect[1] : rect[1]+rect[3], rect[0] : rect[0]+rect[2]] # 2nd crop

def ProcessImage(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Change to greyscale
    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_OTSU) # Apply treshholding
    image = cv2.bitwise_not(image)  # Invert 
    image = cv2.resize(image, img_size, interpolation = cv2.INTER_AREA) # Resize (Networks accept 28x28 images)
    return np.uint8(image) # Change to 8-bit

def PrepareData(image):
    # Prepare data for Neural Network
    TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize(img_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5])
    ])

    pilInput = PIL.Image.fromarray(image) # convert OpenCV image to PIL image
    NNinput = TRANSFORM(pilInput).unsqueeze(0) # convert PIL image to a PyTorch tensor
    data_loader = torch.utils.data.DataLoader(NNinput, batch_size = 1) # Data loader
    
    return data_loader

def DrawResults(image, rectangles, results, value, img_ui_list):
    for i,r in enumerate(rectangles):
        if value[i] > minimal_conf:
            string = str( str(round(value[i]*10000)/100) + "%")
            cv2.putText(image, string, (r[0],r[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0,0,255))

            number = img_ui_list[int(results[i])]
            number = cv2.resize(number, (r[2],r[3]), interpolation = cv2.INTER_AREA)

            if r[0]+r[2] < image.shape[1] and r[1]+r[3] < image.shape[0]:
                #image[ r[1] : r[1]+r[3], r[0] : r[0]+r[2], 2] = number[:,:,2]
                image[ r[1] : r[1]+r[3], r[0] : r[0]+r[2]] = OverlayImage(
                    image[ r[1] : r[1]+r[3], r[0] : r[0]+r[2]], 
                    number
                    )


    return image

def OverlayImage(img1, img2):    
    img1 &= ~img2
    return img1

def PrintResults(results):
    string = str()
    for r in results:
        string += str(r)
    print(string)

img_ui_list = LoadUINumbers(PATH_UI_NUMBERS)

camera = cv2.VideoCapture(0)
if (camera.isOpened()== False):
  print("Error opening video stream or file")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", DEVICE)

network = CNN.Network(n_classes).to(DEVICE) # move network to GPU if available
network.load_state_dict(torch.load(PATH))
network.eval()


while(camera.isOpened()):
    ret, frame_input = camera.read()
    #print(str(frame.shape))

    if SHOW_ALL == True : cv2.imshow('Camera', frame_input)

    key = cv2.waitKey(1)
    if key == 27: break #ESC
    if key == 32: #Space
        while key == 32:
            key = cv2.waitKey(0)

    contours = DetectContours(frame_input)
    rectangles = CreateRectangles(contours, multiplier_contour_size)

    Processed_images = list()

    for rect in rectangles:               
        Processed_images.append(
            ProcessImage(
                CropROI(frame_input, rect)
            )
        )

    prediction_results = list()
    prediction_values = list() 

    for img in Processed_images:     
        input = PrepareData(img)
        prediction = network(next(iter(input)))
        prediction_results.append(int(torch.argmax(prediction))) 
        prediction_values.append(float(torch.max(prediction)))   

    frame_results = DrawRectangles(frame_input, rectangles)
    frame_results = DrawResults(frame_results, rectangles, prediction_results, prediction_values, img_ui_list)
    #PrintResults(prediction_results)
    cv2.imshow('Results', frame_results) 


# After the loop release the cap object
camera.release()
# Destroy all the windows
cv2.destroyAllWindows()