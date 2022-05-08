import numpy as np
import cv2 as cv
import gtts
import playsound
input_lang = 'en-IN'
import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()



haar_cascade = cv.CascadeClassifier(r'C:\Users\Admin\AppData\Local\Programs\Python\Python310\My Programs\Face Recognition\Face-Recognition-System\haarcascade_frontalface_default.xml')

people = ['aamir khan', 'abhishekh  bachchan', 'aishwarya rai bachchan', 'ajay devgn', 'akshay kumar', 'alia bhatt', 'amitabh bachchan', 'anushka sharma', 'arjun kapoor', 'ashnoor kaur', 'avneet kaur', 'deepika padukone','jacqueline fernandez', 'john abraham', 'kajol', 'kareena kapoor', 'karishma kapoor', 'kartik aryan', 'katrina kaif', 'kiara advani', 'kl rahul', 'kriti sanon', 'madhuri dixit', 'malaika arora', 'mohsin khan', 'MS Dhoni', 'neetu kapoor', 'neha kakkar', 'nick jonas', 'parineeti chopra', 'priyanka chopra', 'ranbir kapoor', 'ranveer singh', 'remo desouza', 'ritesh deshmukh', 'rohit sharma', 'salman khan', 'sara ali khan', 'shahid kapoor', 'shahrukh khan', 'shakti mohan', 'shilpa  shetty', 'shivangi joshi', 'shraddha kapoor', 'sidharth malhotra', 'sunny leone', 'sushant singh rajput', 'varun dhawan', 'vicky kaushal', 'virat kohli']


face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
#img=cv.imread(r'C:\Users\Admin\Downloads\ash.jpg')
img = cv.imread(r'C:\Users\Admin\AppData\Local\Programs\Python\Python310\My Programs\Face Recognition\Face-Recognition-System\images\akshay kumar\images2.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')
    speak(f'Label = {people[label]} with a confidence of {confidence}')
    print(people[label])
    people_name=people[label]
    #converted_audio = gtts.gTTS(people_name, lang=input_lang)
    #print('saving audio')
    #converted_audio.save('people.mp3')
    

    cv.putText(img, str(people[label]), (0,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    #resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)

#cv.imshow('Resized', resized)

cv.imshow('Detected Face', img)
#print('playing audio')
#playsound.playsound('people.mp3')

cv.waitKey(0)