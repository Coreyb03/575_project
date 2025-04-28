Name and CSM ID of each member:
Isaiah Stene, 10865968
Corey Becker, 10888204
Zeynep Ankut, 10921218

What programming language is being used:
Python

How the code is structured:
Our code is in a notebook file and is separated into code blocks with labels above them to state what they do. They are split between sections for importing the 
dataset, initial visualization of the features, and extracting and augmenting the data with bootstrapping. Next, we have functions to build the CNN model, create loss functions, 
shifting the data, and find performance metrics. It then builds and fits the model and runs the epochs. Lastly, it plots the results of the metrics and predictions.
There is a demo video in our submission file called demo that also walks through the code.

How to run the code, including very specific compilation instructions, if compilation is needed. Instructions such as "compile using g++" are NOT considered specific:
Since our project is in a notebook file, you will need to run it on something that supports this file type such as VSCode with the Jupyter Notebook extension. 
Then you simply have to run all of the code blocks. 

You may also need to import several python libraries, including:
    - kagglehub
    - tensorflow and keras
    - matplotlib
    - numpy
    - typing
    - re
    - os
    - shutil

We also included the file deleteDataset.py that you can run to remove the dataset after you are done since it is pretty large (you may need to run chmod +x deleteDataset.py).