# ﻿PROGRAM INSTALLATION:

We have install Anaconda Distribution 2022.10 which ships with Python 3.9. It provides meta packages for Python versions 3.7, 3.8, and 3.10 like numpy, matploitlib, math.dist() and pillow which we used in our program. Anaconda Distribution can also support up to three versions of Python at a time. Opencv can be install directly from the anaconda terminal using pip comand.

To install Anaconda on Linux;
Step 1: Update system repositories using the update command
Step 2: Install curl package as it permits fetching the installation script of Anaconda
Step 3: Prepare Anaconda Installer
- Switch to the “/tmp”
- use the curl” command for downloading the Anaconda installer script
Step 4: Install Anaconda
- Using the bash anaconda.sh
Step 5: Activate the Environment settings
- Using the “source ~/.bashrc”

# EXECUTION AND USAGE

The executions were done on spyder from anaconda. 

1- upload the image.csv and edgeHistogram.csv files to the same directory of the execution file or copy the link to the image.  
2- in the load data section of the code, replace the various csv file names with your CSV file names or link to your files.
3- We then run the code on spyder to perform the SVM classification on the data; 
4- First the code loads the data from the files, then convert to numpy array. The arrays are then split into test and train sets then classified and finally evaluated to test how well the classification was done by printing the accuracy, classification report and a graph of results. 


# REQUIREMENTS

- Python 3.7, skitlearn, numpy, CSV and mathploitlib.
