Author: God Bennett ([My legal name was changed from Jordan to God.](https://www.researchgate.net/publication/342328687_Why_I_an_atheist_legally_changed_my_name_to_God))


![Alt-Text](https://github.com/JordanMicahBennett/MASKED-UNSAVED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP/blob/main/god_periocular_clustering_summary.png)

# Click on animated image below to zoom!

![Alt-Text](https://github.com/JordanMicahBennett/MASKED-UNSAVED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP/blob/main/Ai%20Masked%20Unsaved%20Face%20Preview.gif)


# 1) Introduction

Unlike [my prior artificial intelligence app](https://github.com/JordanMicahBennett/MASKED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP) that I decided to create because I couldn't find any ai app that did masked face recognition (given that I would save or train the ai model on the target's face beforehand), this solution does not need to save or be trained on the person in mask beforehand to be detected afterwards.

This artificial intelligence project is a masked face recognition (or periocular recognition) artificial intelligence application, built atop pyfacy, that takes a 128 dimension face embedder [from my earlier ai project](https://github.com/JordanMicahBennett/MASKED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP).

This solution was also created last year in November 2020, shortly after [the first solution](https://github.com/JordanMicahBennett/MASKED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP), and now the code public here.

* See crucial modifications and addditions (of the original [standard python pyfacy library](https://pypi.org/project/pyfacy/)) by God below in section (2) "God's crucial modifications discussion and results".
The [original library](https://github.com/ManivannanMurugavel/pyfacy) does not do masked face recognition/clustering. 

* My [earlier masked face recognition ai app](https://github.com/JordanMicahBennett/MASKED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP) needed to have to have a person's face saved before-hand, in order to recognize a person afterwards. This current solution does not require a person's face to be known beforehand.

* The 128 dimensional embedder for masked face recognition was taken from my earlier "[MASKED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP](https://github.com/JordanMicahBennett/MASKED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP)" project, and implemented into pyfacy for unsaved face classification.



# Example use case: Example use case: A case in a bank where we don't have the target's face before hand, but we want to count the number of times the person uses the atm, to for eg flag irregular atm usage count.





# 2) God's crucial modifications discussion and results


PART I)

Both (1) my atm masked face recognition solution "[MASKED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP](https://github.com/JordanMicahBennett/MASKED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP)" and 
(2) [my atm masked unsaved face image counting solution](https://github.com/JordanMicahBennett/MASKED-UNSAVED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP), namely using the pyfacy library (which by default is partially usable for the new atm usage count solution), take face images as inputs, and convert them to embeddings aka numerical representations before performing either the task from (1) or (2).


PART II)


However, pyfacy used in Part I (2) does not do masked face computing well out of the box, unlike the project I had configured in (1).

For example, for the dataset of clustering 3 input identities in this project directory (13 images total), pyfacy produces only 2 face clusters, completely ignoring the images for Holness' face with masked images.

A) This latest machine learning modification of pyfacy (2) by God essentially replaces pyfacy's default face embedding function, 
with an external function adapted from (1) that generates embeddings of only a portion of the face, which roughly excludes the mask. 

B) To do (A), since masked processing in (1) worked well, of the 4 python files from (1):

i) [extract_embeddings.py](https://github.com/JordanMicahBennett/MASKED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP/blob/main/extract_embeddings.py) is duplicated. 

ii) extract_embeddings.py [is modified to process a single image at a time](https://github.com/JordanMicahBennett/MASKED-UNSAVED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP/blob/main/GOD_MODIFIED_PYFACY_PYTHON_LIBRARY_FILES/external_extract_embeddings_pyfacy_god.py) (instead of the batch that it does normally), and generate one single embedding.

iii) that duplication is then placed into pyfacy's python root library in site packages, and imported in pyfacy's util python file, which modifies the "load_images_to_clust_encodings" function which takes the embedding discussed in Part II.

Other modifications to facilitate the above include modifying the main pyfacy demo, to accept machine learning model source directory associated with masked face embedding:

(1) %AppData%.../Programs/Python36/Lib/site-packages/pyfacy/utils.py "load_images_to_clust_encodings" modified to both run face embedding from external source (masked face embedder), as well as machine learning model source associated with external embedder

(2) %AppData%.../Programs/Python36/Lib/site-packages/pyfacy/face_clust/algorithm.py "load_faces" modified to both run face embedding from external source (masked face embedder), as well as machine learning model source associated with external embedder

(3) (1) and (2) results in the change here. Much more detailed change list outlined in Instructions.txt.




# 3) God's Instructions

1. Install python 3.6.

2. Install pyfacy==1.0.1

* Note if you get dlib build red line errors, you need to:

   * Install [cmake 3.19.4](https://cmake.org/download/), and [visual studio build tools](https://go.microsoft.com/fwlink/?LinkId=691126)
   * Install pyfacy again using both cmake bin and python36 directory in path.

3. Modify the installation files, by copying the files from "GOD_MODIFIED_PYFACY_PYTHON_LIBRARY_FILES" to Python36/Lib/site-packages/pyfacy directory (normally in %AppData%.../Programs/Python36)
Of the 2 files and folder copied above, you should accept prompts to replace the utils.py file, as well as the face_clust folder content "algorithm.py".

4. Crucially for gaining the ability to do masked face recognition in pyfacy 1.0.1:

* i. Download the [.t7 open face model](https://github.com/JordanMicahBennett/MASKED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP#a-instructions_user) from my "[MASKED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP](https://github.com/JordanMicahBennett/MASKED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP).

* ii. Download the [face detection model](https://github.com/JordanMicahBennett/MASKED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP/tree/main/face_detection_model) from my "[MASKED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP](https://github.com/JordanMicahBennett/MASKED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP).

* iii. Place both the folder from (i) and the .t7 file from (iii) in one folder somewhere.

* iv. Modify the parameter of mdl.load_faces in runClusteringTool.py to be the one you created from above.





# 4) Visualizing the necessary/required difference between default pyfacy, and pyfacy modified by God

	
To see the difference between default pyfacy and God's pyfacy, discussed in (2): "God's crucial modifications discussion and results":

----Run default

i) Go to %AppData%.../Programs/Python36/Lib/site-packages/pyfacy/utils.py "load_images_to_clust_encodings", comment out my line (144), and uncomment default line (142).

ii) If cluster output folder exists in this project dir, delete it.

ii) Run runClusteringTool.py to run the project. Notice only folders for 2 faces are produced in output/, although 3 input identities (13 images total) were supplied. The masked face was not found.

----Run God's modification

i) Go to %AppData%.../Programs/Python36/Lib/site-packages/pyfacy/utils.py "load_images_to_clust_encodings", uncomment my line (144), and comment out default line (142).

ii) If cluster output folder exists in this project dir, delete it.

ii) Run runClusteringTool.py to run the project. Notice folders for all 3 faces are produced in output/, correlating correctly with 3 input identities supplied in dataset. The masked face was successfully found.




# 5) Why this new clustering solution is reasonably needed.

Data: Each simulated atm image signifies an identity with 1 atm usage event.

Result: God has framed the task of identifying multiple atm usages per identity (the latest requirement from fraud team), as a clustering problem/solution. 

•	The machine learning algorithm is able to cluster faces from a pool of atm images,  that is, it “simply” sorts the face images as clusters aka groups. 
	
	o	Each resulting cluster pertains to 1 identity.
	
	o	Any resulting cluster with more than 1 image, signifies multiple atm usage for that identity.

	
# 6) Components and running the app (basic)

2) [LAUNCH_APP.bat](https://github.com/JordanMicahBennett/MASKED-UNSAVED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP/blob/main/LAUNCH_APP.bat) - Machine learning/Artificial Intelligence Clustering/Grouping algorithm. 

[Run LAUNCH_APP.bat. Wait on resulting “output/” directory to be generated. Delete the output directory before running the batch file for other cases. ]

1) Customized Machine learning/Artificial Intelligence input embedding handler. [This is ran automatically in (2) above]





	
# 6) Components and running the app (advanced)

1) Jpeg converter to convert input png images to jpeg format used by this particular machine learning model. [Too add png images from atm to dataset as jpeg, run jpg_converter.py with Python idle. Adjust new directory name by adjusting name at the end of value of individualPath variable top of program, then copy resulting jpeg images to dataset directory. ]

2) Customized Machine learning/Artificial Intelligence input embedding handler. [This is ran automatically in (3) below]

3) [LAUNCH_APP.bat](https://github.com/JordanMicahBennett/MASKED-UNSAVED-FACE-RECOGNITION-OR-PERIOCULAR-ARTIFICIAL-INTELLIGENCE-APP/blob/main/LAUNCH_APP.bat) - Machine learning/Artificial Intelligence Clustering/Grouping algorithm. 

[Run LAUNCH_APP.bat. Wait on resulting “output/” directory to be generated. Delete the output directory before running the batch file for other cases. ]

	
# Suggested Training Data
See [the famous Masked Dataset paper](https://arxiv.org/abs/2003.09093), that unfortunately doesn't provide a solution/code, but it does provide a gigabytes of data for training. Neither of my solutions use this currently, and both could reasonably benefit from the aforementioned training set. There are also solutions for creating masked data available across the web.
	
