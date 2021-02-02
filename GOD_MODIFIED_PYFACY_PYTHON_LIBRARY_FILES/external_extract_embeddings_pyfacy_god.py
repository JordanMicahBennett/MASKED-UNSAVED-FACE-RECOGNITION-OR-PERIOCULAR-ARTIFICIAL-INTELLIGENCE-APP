#The code below added by God to pyfacy's internal libraries, for masked face recognition.

# import libraries
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from os.path import dirname, join


#modified by God to return a single face embedding, based on a new imagePath parameter
def getFaceEmbedding (imagePath, externalFaceEmbeddingModelDirectory):
        # load serialized face detector
        print("Loading Face Detector...")
        ___mainLocationDir___ = externalFaceEmbeddingModelDirectory
        
        
        ___modelMainLocationDir___ = ___mainLocationDir___ + "/face_detection_model/"

        protoPath = join(dirname(___modelMainLocationDir___), "deploy.prototxt")
        modelPath = join(dirname(___modelMainLocationDir___), "res10_300x300_ssd_iter_140000.caffemodel")
        detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

        # load serialized face embedding model
        print("Loading Face Recognizer...")
        embedder = cv2.dnn.readNetFromTorch(join(dirname(___mainLocationDir___), "openface_nn4.small2.v1.t7"))
        

        # initialize the total number of faces processed
        total = 0

        ##################
        # God edit: loop removed, in place of single image processor

        # load the image, resize it to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # known embeddings
        knownEmbeddings = [ ]

        # ensure at least one face was found
        if len(detections) > 0:
                # we're making the assumption that each image has only ONE face, so find the bounding box with the largest probability
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]

                # ensure that the detection with the largest probability also means our minimum probability test (thus helping filter out weak detections)
                if confidence > 0.5:
                        # compute the (x, y)-coordinates of the bounding box for the face
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")


                        # extract the face ROI and grab the ROI dimensions
                        # God modification: On R.H.S below, "endY" changed to "int(endY-(.27*endY))",
                        # ...to focus only on extracting all but last 27% of face (i.e. all but roughly region of mask vertically
                        face = image[startY:int(endY-(.27*endY)), startX:endX]
                        #face = image[startY:endY, startX:endX]
                        (fH, fW) = face.shape[:2]

                        # construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
                        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                (96, 96), (0, 0, 0), swapRB=True, crop=False)
                        embedder.setInput(faceBlob)
                        vec = embedder.forward()
                        knownEmbeddings.append(vec.flatten())


        return knownEmbeddings[0] if len(knownEmbeddings) > 0 else None
      

			
