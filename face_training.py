import os
import numpy as np
import cv2 as cv

people = ['aamir khan', 'abhishekh  bachchan', 'aishwarya rai bachchan', 'ajay devgn', 'akshay kumar', 'alia bhatt', 'amitabh bachchan', 'anushka sharma', 'arjun kapoor', 'ashnoor kaur', 'avneet kaur', 'deepika padukone', 'jacqueline fernandez', 'john abraham', 'kajol', 'kareena kapoor', 'karishma kapoor', 'kartik aryan', 'katrina kaif', 'kiara advani', 'kl rahul', 'kriti sanon', 'madhuri dixit', 'malaika arora', 'mohsin khan', 'MS Dhoni', 'neetu kapoor', 'neha kakkar', 'nick jonas', 'parineeti chopra', 'priyanka chopra', 'ranbir kapoor', 'ranveer singh', 'remo desouza', 'ritesh deshmukh', 'rohit sharma', 'salman khan', 'sara ali khan', 'shahid kapoor', 'shahrukh khan', 'shakti mohan', 'shilpa  shetty', 'shivangi joshi', 'shraddha kapoor', 'sidharth malhotra', 'sunny leone', 'sushant singh rajput', 'varun dhawan', 'vicky kaushal', 'virat kohli']
DIR = r'C:\Users\Admin\AppData\Local\Programs\Python\Python310\My Programs\Face Recognition\Face-Recognition-System\images'

haar_cascade = cv.CascadeClassifier(r'C:\Users\Admin\AppData\Local\Programs\Python\Python310\My Programs\Face Recognition\Face-Recognition-System\haarcascade_frontalface_default.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            if img_array is None:
                continue 
                
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done for the Model---------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
