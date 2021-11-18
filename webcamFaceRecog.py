import face_recognition
import cv2
from face_recognition.api import face_distance, face_encodings, face_locations
import numpy as np

video_capture=cv2.VideoCapture(0)

antal_image=face_recognition.load_image_file("antal.jpg")
antal_encoding= face_recognition.face_encodings(antal_image)[0]
smara_image=face_recognition.load_image_file("smara.jpg")
smara_encoding=face_recognition.face_encodings(smara_image)[0]
florin_image=face_recognition.load_image_file("florin.jpg")
florin_encoding=face_recognition.face_encodings(florin_image)[0]

known_face_encodings= [
    smara_encoding,
    antal_encoding,
    florin_encoding
]
known_face_names=[
    "Smara",
    "Antal",
    "Florin"
]
known_face_info= [
    "Simp",
    "Ieeia,21",
    "Gay"
]

face_locations = []
face_encodings= []
face_names= []
face_infos=[]
process_this_frame= True
while True:
    ret,frame= video_capture.read()
    
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    
    rgb_small_frame=small_frame[:,:,::-1]
    
    if process_this_frame:
        face_locations= face_recognition.face_locations(rgb_small_frame)
        face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)
        
        face_names=[]
        face_infos=[]
        for face_encoding in face_encodings:
            matches=face_recognition.compare_faces(known_face_encodings,face_encoding)
            name="Unknown"
            info="Unknown"
            face_distances= face_recognition.face_distance(known_face_encodings,face_encoding)
        
            best_match_index=np.argmin(face_distance)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                info=known_face_info[first_match_index]
            # if(matches[best_match_index]):
            #     name=known_face_names[best_match_index]
            #     info=known_face_info[best_match_index]
        
            face_names.append(name)
            face_infos.append(info)
    process_this_frame= not process_this_frame
    
    #display reslts
    
    
    for(top,right,bottom,left),name,info in zip(face_locations,face_names,face_infos):
        top*=4
        right*=4
        bottom*=4
        left*=4
        #draw a box around the face
        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
        
        #draw a label with name
        cv2.rectangle(frame,(left,bottom-35),(right,bottom),(0,0,255),cv2.FILLED)
        font=cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame,name,(left+6,top -6),font ,1.0,(255,255,255),1)
        cv2.putText(frame,info,(left+6,bottom-6),font ,1.0,(255,255,255),1)
        
    cv2.imshow('Video',frame)
    
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break
  
  
video_capture.release()  
cv2.destroyAllWindows()