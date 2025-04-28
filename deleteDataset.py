import kagglehub
import os
import shutil



file_path = "~/.cache/kagglehub/datasets/fantineh/"
user_path = os.path.expanduser(file_path)

list = os.listdir(user_path)
if(len(list) != 0):
    print("deleting wildfire dataset")
    # os.rmdir(user_path)
    shutil.rmtree(user_path + 'next-day-wildfire-spread/')
else:
    print("already deleted")
