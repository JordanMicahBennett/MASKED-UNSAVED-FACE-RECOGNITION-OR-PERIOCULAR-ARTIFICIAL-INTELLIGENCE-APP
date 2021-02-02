#Modified by God to supply directory to face embedding model
# (1) %AppData%.../Programs/Python36/Lib/site-packages/pyfacy/utils.py "load_images_to_clust_encodings" modified to both run face embedding from external source (masked face embedder), as well as machine learning model source associated with external embedder
# (2) %AppData%.../Programs/Python36/Lib/site-packages/pyfacy/face_clust/algorithm.py "load_faces" modified to both run face embedding from external source (masked face embedder), as well as machine learning model source associated with external embedder
# (3) (1) and (2) results in the change here. Much more detailed change list outlined in Instructions.txt.


from pyfacy import face_clust
import os

mdl = face_clust.Face_Clust_Algorithm("dataset/")

#Parameter=saved machine learning model directory
mdl.load_faces("C:/Users/Jordan/Downloads/God/RobotizeJa/Facial Recognition/Masked Face Recognition Files/Masked Face Recognition Artificial Intelligence by God")

os.mkdir ('output/')

mdl.save_faces('output/')
