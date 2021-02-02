# Author: God Bennett
# Needed because pyfacy relies on jpg format
#Notes: Just change individualPath end/person name to generate personn data
from PIL import Image
import difflib

def getStringDifference(stringA, stringB):
	output_list = [li for li in difflib.ndiff(stringA, stringB) if li[0] != ' ']
	output_string = ''.join(str(ele) for ele in output_list)
	output_string = output_string.replace ("+ ", "")
	return output_string

import os
mainPath = 'C:/Users/bennettjm/Documents/___Tasks/face_atm_detection/counting_strategy/ncb_god_pyfacy_py3.6_atm_face_recognition_cluster/'
individualPath = 'C:/Users/bennettjm/Documents/___Tasks/face_atm_detection/counting_strategy/ncb_god_pyfacy_py3.6_atm_face_recognition_cluster/aakash'
individualName = getStringDifference(mainPath, individualPath)

files = os.listdir(individualPath)

os.mkdir(mainPath + "jpeg/" )
os.mkdir(mainPath + "jpeg/" + individualName )

for index, file in enumerate(files):
    fileName = os.path.join(individualPath, file)
    im = Image.open(fileName)
    rgb_im = im.convert('RGB')
    rgb_im.save(mainPath + "jpeg/" + individualName + "/" + str(index) + ".jpg")


