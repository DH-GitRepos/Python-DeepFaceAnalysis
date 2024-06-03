# DEMO FOR COMP60037: WEB AND AI (Assessment 3)
# Demo by Darren Halpin (21022839)

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from deepface import DeepFace as df # pip install deepface, also, pip install tf-keras
import pandas as pd
import os
import glob
import cv2 as cv

backends = [        # CNN MODELS FOR DETECTION TASKS
  'opencv',         # [0] faster
  'ssd',            # [1] faster
  'mtcnn',          # [2] slower, but more accurate
  'retinaface',     # [3] slower, but more accurate
  'mediapipe',      # [4]
  'yolov8',         # [5] pip install ultralytics
  'yunet',          # [6]
  'fastmtcnn',      # [7] pip install facenet-pytorch
  'dlib'            # [8]
]

models = [          # FR MODELS
  "VGG-Face",       # [0] 4096 data points (default) (Oxford Uni)
  "Facenet",        # [1] 128 data points (Google)
  "Facenet512",     # [2] 521 data points (BEST PERFORMING @ 99.65%)
  "OpenFace",       # [3] 128 data points (Carnegie Mellon Uni)
  "DeepFace",       # [4] 4096 data points (Facebook)
  "DeepID",         # [5] 160 data points
  "ArcFace",        # [6] 512 data points
  "SFace",          # [7] 128 data points
]

metrics = [         # SIMILARITY CALCULATION METRICS FOR FACE RECOGNITION
    "cosine",       # (default)
    "euclidean",
    "euclidean_l2"  # Best (most stable) metric, with the Facenet512 model
]

file_path = os.path.dirname(os.path.abspath(__file__))
img_path = f"{file_path}\\Images"
img_categories_path = f"{img_path}\\Sample-sets"

# create image path arrays for each emotion category
faces_set_1 = glob.glob(f"{img_categories_path}\\Set-01\\*.png")
faces_set_2 = glob.glob(f"{img_categories_path}\\Set-02\\*.png")
faces_set_3 = glob.glob(f"{img_categories_path}\\Set-03\\*.png")
faces_set_4 = glob.glob(f"{img_categories_path}\\Set-04\\*.png")

# create dictionary of emotion categories
emotion_images = {
    "set-1": faces_set_1,
    "set-2": faces_set_2,
    "set-3": faces_set_3,
    "set-4": faces_set_4,
}
        
# FACIAL ANALYSIS (image sets)
for image in emotion_images['set-4']:

  demographies = df.analyze(img_path=image,                          
                            enforce_detection=True,
                            detector_backend=backends[8],  # [3] retinaFace, [4] mediapipe, [0] openCV
                            align=True,
                            actions=['emotion']                                
                            )

  for person in demographies:
      
    filename = os.path.basename(image)  
    print(f"IMAGE: {filename}")
    print(f"EMOTION: {person['dominant_emotion']} :"
          f" {person['emotion'][person['dominant_emotion']]}")
    print("------------")


# FACIAL ANALYSIS (single image)
# - actions: tuple | list = ("emotion", "age", "gender", "race"),
# demographies = df.analyze(img_path=emotion_images['set-4'][0],                          
#                           enforce_detection=True,
#                           detector_backend=backends[3],  # [3] retinaFace
#                           align=True
#                           )

# # print(len(demographies))
# for person in demographies:
    
#   print(f"GENDER: {person['dominant_gender']} :"
#         f" {person['gender'][person['dominant_gender']]}")
#   print(f"AGE: {person['age']}")
#   print(f"RACE: {person['dominant_race']} :"
#         f" {person['race'][person['dominant_race']]}")
#   print(f"EMOTION: {person['dominant_emotion']} :"
#         f" {person['emotion'][person['dominant_emotion']]}")
#   print("------------")

# objs = df.analyze(img_path = images[4],
#         actions = ['age', 'gender', 'race', 'emotion']
# )


# LIVE FACIAL ANALYSIS (USING WEBCAM)
# def stream():
#     cap = cv.VideoCapture(0)  # 0 is the default webcam

#     while True:
#         ret, frame = cap.read()  # Capture frame-by-frame
#         if not ret:
#             break

#         result = df.analyze(frame, 
#                             actions=['emotion'],
#                             detector_backend=backends[0])  # [0] openCv, [4] mediapipe, [2] mtcnn)

#         dominant_emotion = result[0]['dominant_emotion']
#         cv.putText(frame, dominant_emotion, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

#         cv.imshow('frame', frame)

#         if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
#             break

#     cap.release()
#     cv.destroyAllWindows()

# stream()
