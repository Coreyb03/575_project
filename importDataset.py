import kagglehub
import os

file_path = "~/.cache/kagglehub/datasets/fantineh/next-day-wildfire-spread/versions/2"

# Download latest version
if not os.path.exists(file_path):
    path = kagglehub.dataset_download("fantineh/next-day-wildfire-spread")
    print("Path to dataset files:", path)
else:
    print("file found at ", file_path )


