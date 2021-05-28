# Abnormality-detection-using-Deep-Learning-Techniques
Detecting abnormalities in bones using deep laerning

Dataset used- MURA(musculoskeletal radiographs)
positive- abnormal XRAY's and negetive- normal XRAY's
Aim- Analyze XRAY's of the shoulder, humerus, elbow, forearm, wrist, hand, and finger and classify them into normal or abnormal.

Used many deep learning models to compare their accuracy on the MURA dataset. ResNet gave the highest accuracy.

data folder contains the train data set for the model divided into negative and positive folders
models folder contains different models used
train.py file loads the model you want to use in the first line and saves that model, also giving accuracy of the model used
there is also a log to keep track on epoch,acc,loss,val_acc,val_loss. This helps in tuning hyperparameters of the deep learning model

This project can automate abnormality localisation and improve the workflow of radiologists, also helping the radiologist to confirm their diagnosis.
