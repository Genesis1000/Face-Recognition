####################################################
# Modified by Sijo Jacob &  Alvin  Aexander        # 
# Original code: http://thecodacus.com/            #
# All right reserved to the respective owner       #
####################################################
from Tkinter import*
#import wx
import Tkinter
import tkMessageBox
import cv2
import os
import numpy as np
from PIL import  ImageTk,Image

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
top = Tkinter.Tk()

# Give the window a title.
top.title("Face Recognizer")
# Change the minimum size.
top.minsize(500, 500)
# Change the background colour.
top.configure(bg="#67BCDB")

E1 = Entry(top, bd =3)
E1.grid(row=6,column=3)
L1 = Label(top, text="enter your name : ").grid(row=6,column=2)
L2 = Label(top, text="WELCOME TO FACE RECOGNITION : ").grid(row=1,column=3)
def dataset():
    # Start capturing video 
    vid_cam = cv2.VideoCapture(0)

    # Detect object in video stream using Haarcascade Frontal Face
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    count = 0

    assure_path_exists("datasets/")

    # Start looping
    while(True):

        # Capture video frame
        _, image_frame = vid_cam.read()
        bl=image_frame

        # Convert frame to grayscale
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

        # Detect frames of different sizes, list of faces rectangles
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        # Loops for each faces
        for (x,y,w,h) in faces:

            # Crop the image frame into rectangle
            cv2.rectangle(image_frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
            # Increment sample face image
            count += 1
            data=Entry.get(E1)

            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + data + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            cv2.imwrite("originalcolor/User." + data + '.' + str(count) + ".jpg", bl)

            # Display the video frame, with bounded rectangle on the person's face
            cv2.imshow('frame', image_frame)

        # To stop taking video, press 'q' for at least 100ms
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        # If image taken reach 100, stop taking video
        elif count>30:
            break

    # Stop video
    vid_cam.release()

    # Close all started windows
    cv2.destroyAllWindows()
def training():
    vid_cam = cv2.VideoCapture(0)
    # Create Local Binary Patterns Histograms for face recognization
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Using prebuilt frontal face training model, for face detection
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

    # Create method to get the images and label data
    arr=[]
    n=[]
    def getImagesAndLabels(path):
        # Get all file path
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
        
        # Initialize empty face sample
        faceSamples=[]
        
        # Initialize empty id
        ids = []
        
        valn=1
        i=0

        # Loop all the file path
        for imagePath in imagePaths:

            # Get the image and convert it to grayscale
            PIL_img = Image.open(imagePath).convert('L')
            

            # PIL image to numpy array
            img_numpy = np.array(PIL_img,'uint8')

            # Get the image id
            id = str(os.path.split(imagePath)[-1].split(".")[1])
             
            # Get the face from the training images
            faces = detector.detectMultiScale(img_numpy)
            

            # Loop for each face, append to their respective ID
            for (x,y,w,h) in faces:
                i=i+1

                # Add the image to face samples
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                arr.append(id)
                #print(arr)
                if i==32:
                 valn=valn+1
                 i=0
                 
                n.append(valn)

                #traning data 
                #cv2.imshow("Adding faces to traning set...", img_numpy[y: y + h, x: x + w])
                #cv2.waitKey(10)
                    
                
        # Pass the face array and IDs array
        return faceSamples,n
    
    # Get the faces and IDs
    faces,n = getImagesAndLabels('dataset')
    
    #print(n)
    
     # Train the model using the faces and IDs
    recognizer.train(faces,np.array(n))    # just chage the data's in Id to int remove the error

    # Save the model into trainer.yml
    assure_path_exists('trainer/')
    recognizer.save('trainer/trainer.yml')
#def facerecognition():
    # Create Local Binary Patterns Histograms for face recognization
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    assure_path_exists("trainer/")


    # Load the trained mode
    recognizer.read('trainer/trainer.yml')

    # Load prebuilt model for Frontal Face
    cascadePath = "haarcascade_frontalface_default.xml"

    # Create classifier from prebuilt model
    faceCascade = cv2.CascadeClassifier(cascadePath);

    # Set the font style
    font = cv2.FONT_HERSHEY_SIMPLEX
    

    # Initialize and start the video frame capture
    cam = cv2.VideoCapture(0)
    #print n
    b=0
    # Loop
    while True:
        # Read the video frame
        ret, im =cam.read()

        # Convert the captured frame into grayscale
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

        # Get all face from the video frame
        faces = faceCascade.detectMultiScale(gray, 1.2,5)
        

        # For each face in faces
        for(x,y,w,h) in faces:

            # Create rectangle around the face
            cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,0,255), 4)

            # Recognize the face belongs to which ID
            Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            a=int(Id)
            #print confidence
            #print arr[a*30]
            
           
            if confidence>40:
            #Id = "{ 1:.2f}%".format(round(50 - confidence, 2))

            # Put text describe who is in the picture
             cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (45,35,100), -1)
             c=arr[a*28].upper()
             d=cv2.putText(im,c, (x,y-40), font, 1, (255,255,255), 2)
             vid_cam.read()
             b=b+1
             cv2.imwrite("color/User." +arr[a*28] +str(b)+ ".jpg",d )
            
            else:
             cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
             cv2.putText(im, "UNKNOWN", (x,y-40), font, 1, (255,255,255), 3)   

        # Display the video frame with the bounded rectangle
        cv2.imshow('Press q to exit',im)
        
        # If 'q' is pressed, close program
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Stop the camera
    cam.release()

    # Close all windows
    cv2.destroyAllWindows()


B2=Button(top, text ="Recognize Your Face",width=40,command = training,fg="red").grid(row=14,column=3)
#B3=Button(top, text ="Recognise Face",command = facerecognition,fg="blue").grid(row=2,column=3)
B1=Button(top, text ="Create the dataSet",width=40,command = dataset,fg="green").grid(row=10,column=3)
#B4=Button(top,text='Ratings',width=40).grid(row=16,column=3)

#import optionlisting 
top.mainloop()


    
