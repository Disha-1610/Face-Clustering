# Unsupervised Face-Clustering Pipeline
Distributed Video Face Clustering Pipeline using Unsupervised Learning 
A scalable computer vision system that processes video footage, extracts facial embeddings, and automatically groups similar faces using clustering techniques. Designed with multithreaded processing and production-ready job-based architecture.

##🚀Features

🎞️ Video frame extraction
🧑 Face detection using deep learning models
🔢 Face embedding generation
🧩 Unsupervised face clustering using DBSCAN
⚡ Multithreaded processing pipeline
📂 Automatic grouped face storage
🖼️ Face montage generation
🗂️ Job-based output architecture
📊 Logging and error monitoring

##🛠️Tech Stack

Python
OpenCV
NumPy
dlib / face_recognition
Scikit-learn
pyPiper (Multithreading Pipeline)
tqdm (Progress tracking)
Logging (Monitoring)

##⚙️ Installation

 Dependencies
```bash
pip install opencv-python numpy scikit-learn dlib face_recognition imutils pyPiper tqdm
```
##⚙️ ▶️ Usage

 ```bash

```

##🧩Clustering Method

This project uses DBSCAN clustering, which:
Does not require labeled data
Automatically detects number of unique individuals
Groups faces based on embedding similarity

##📊 Output Example

Each folder represents one detected identity cluster.
```bash
ClusteredFaces/
   ├── Face_0/
   ├── Face_1/
   ├── Face_2/
   └── Montage/
```
##💼 Real World Applications

Surveillance analytics
Event media tagging
Identity grouping in video datasets
Dataset creation for supervised training
Security and monitoring systems



