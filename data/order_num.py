import os

path = "facebank" + os.sep + "normal"
dirs = os.listdir(path)

i = 1
for file_name in dirs:
   os.rename(path + os.sep + file_name, path + os.sep + str(i) + ".jpg")
   i += 1
