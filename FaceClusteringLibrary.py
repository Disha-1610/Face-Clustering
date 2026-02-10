
from sklearn.cluster import DBSCAN
from imutils import build_montages, paths
import numpy as np
import os
import pickle
import cv2
import shutil
import time
import dlib
from pyPiper import Node, Pipeline
from tqdm import tqdm
import face_recognition

''' Common utilities '''
'''
The ResizeUtils provides resizing function to keep the aspect ratio intact
'''
class ResizeUtils:
    # Given a target height, adjust the image by calculating the width and resize
    def rescale_by_height(self, image, target_height, method=cv2.INTER_LANCZOS4):
        """Rescale `image` to `target_height` (preserving aspect ratio)."""
        w = int(round(target_height * image.shape[1] / image.shape[0]))
        return cv2.resize(image, (w, target_height), interpolation=method)

    # Given a target width, adjust the image by calculating the height and resize
    def rescale_by_width(self, image, target_width, method=cv2.INTER_LANCZOS4):
        """Rescale `image` to `target_width` (preserving aspect ratio)."""
        h = int(round(target_width * image.shape[0] / image.shape[1]))
        return cv2.resize(image, (target_width, h), interpolation=method)



''' Frames extractor from video footage '''
'''
The FramesGenerator extracts image frames from the given video file
The image frames are resized for dlib processing
'''
class FramesGenerator:
    def __init__(self, VideoFootageSource):
        self.VideoFootageSource = VideoFootageSource

    def AutoResize(self, frame):
        resizeUtils = ResizeUtils()
        height, width, _ = frame.shape

        if height > 500:
            frame = resizeUtils.rescale_by_height(frame, 500)
            return self.AutoResize(frame)

        if width > 700:
            frame = resizeUtils.rescale_by_width(frame, 700)
            return self.AutoResize(frame)

        return frame

    def GenerateFrames(self, OutputDirectoryPath, frame_interval=1):
        cap = cv2.VideoCapture(self.VideoFootageSource)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        os.makedirs(OutputDirectoryPath, exist_ok=True)

        logging.info(f"Video FPS: {fps}, Total Frames: {total_frames}")

        frame_step = int(fps * frame_interval)
        frame_count = 0
        saved_count = 1

        while frame_count < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()

            if not ret or frame is None:
                logging.warning(f"Failed to read frame {frame_count}")
                frame_count += frame_step
                continue

            frame = self.AutoResize(frame)

            filename = f"frame_{saved_count}.jpg"
            cv2.imwrite(os.path.join(OutputDirectoryPath, filename), frame)

            saved_count += 1
            frame_count += frame_step

        cap.release()
        logging.info("Frames extraction finished")

        

''' Face clustering multithreaded pipeline '''
'''
Following are nodes for pipeline constructions. It will create and asynchronously
execute threads for reading images, extracting facial features and storing 
them independently in different threads
'''
# Keep emitting the filenames into the pipeline for processing
class FramesProvider(Node):
    def setup(self, sourcePath):
        self.sourcePath = sourcePath
        self.filesList = []
        for item in os.listdir(self.sourcePath):
            _, fileExt = os.path.splitext(item)
            if fileExt == '.jpg':
                self.filesList.append(os.path.join(item))
        self.TotalFilesCount = self.size = len(self.filesList)
        self.ProcessedFilesCount = self.pos = 0

    # Emit each filename in the pipeline for parallel processing
    def run(self, data):
        if self.ProcessedFilesCount < self.TotalFilesCount:
            self.emit({'id': self.ProcessedFilesCount, 
                'imagePath': os.path.join(self.sourcePath, 
                                self.filesList[self.ProcessedFilesCount])})
            self.ProcessedFilesCount += 1
            
            self.pos = self.ProcessedFilesCount
        else:
            self.close()

# Encode the face embedding, reference path and location 
# and emit to pipeline
class FaceEncoder(Node):
    def setup(self, detection_method = 'cnn'):
        self.detection_method = detection_method
        # detection_method can be cnn or hog

    def run(self, data):
        id = data['id']
        imagePath = data['imagePath']
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model=self.detection_method)
        encodings = face_recognition.face_encodings(rgb, boxes)
        d = [{"imagePath": imagePath, "loc": box, "encoding": enc} 
                for (box, enc) in zip(boxes, encodings)]

        self.emit({'id': id, 'encodings': d})

# Recieve the face embeddings for clustering and 
# id for naming the distinct filename
class DatastoreManager(Node):
    def setup(self, encodingsOutputPath):
        self.encodingsOutputPath = encodingsOutputPath
    def run(self, data):
        encodings = data['encodings']
        id = data['id']
        with open(os.path.join(self.encodingsOutputPath, 
                            'encodings_' + str(id) + '.pickle'), 'wb') as f:
            f.write(pickle.dumps(encodings))

# Inherit class tqdm for visualization of progress
class TqdmUpdate(tqdm):
    # This function will be passed as progress callback function
    # Setting the predefined variables for auto-updates in visualization
    def update(self, done, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.n = done
        super().refresh()



''' Pickle files merging '''
'''
PicklesListCollator takes multiple pickle files as input and merge them together
It is made specifically to support our use-case of merging distinct pickle
files into one
'''
class PicklesListCollator:
    def __init__(self, picklesInputDirectory):
        self.picklesInputDirectory = picklesInputDirectory

    def GeneratePickle(self, outputFilepath):
        datastore = []

        pickle_files = [
            os.path.join(self.picklesInputDirectory, f)
            for f in os.listdir(self.picklesInputDirectory)
            if f.endswith(".pickle")
        ]

        for picklePath in pickle_files:
            try:
                with open(picklePath, "rb") as f:
                    data = pickle.loads(f.read())
                    datastore.extend(data)
            except Exception as e:
                logging.error(f"Failed reading {picklePath}: {e}")

        with open(outputFilepath, "wb") as f:
            pickle.dump(datastore, f)

        logging.info("Merged pickle created")




''' Face clustering functionality '''
class FaceClusterUtility:
    def __init__(self, EncodingFilePath, eps=0.5):
        self.EncodingFilePath = EncodingFilePath
        self.eps = eps

    def Cluster(self):

        if not os.path.isfile(self.EncodingFilePath):
            raise FileNotFoundError("Encoding pickle not found")

        logging.info("Loading encodings")
        data = pickle.loads(open(self.EncodingFilePath, "rb").read())
        data = np.array(data)

        encodings = [d["encoding"] for d in data]

        logging.info("Running DBSCAN clustering")
        clt = DBSCAN(eps=self.eps, metric="euclidean", n_jobs=-1)
        clt.fit(encodings)

        labelIDs = np.unique(clt.labels_)
        numUniqueFaces = len(np.where(labelIDs > -1)[0])

        logging.info(f"Unique faces detected: {numUniqueFaces}")

        return clt.labels_


class FaceImageGenerator:
    def __init__(self, EncodingFilePath):
        self.EncodingFilePath = EncodingFilePath

    def GenerateImages(self, labels, OutputFolderName="ClusteredFaces", MontageOutputFolder="Montage"):

        os.makedirs(OutputFolderName, exist_ok=True)

        MontageFolderPath = os.path.join(OutputFolderName, MontageOutputFolder)
        os.makedirs(MontageFolderPath, exist_ok=True)

        data = pickle.loads(open(self.EncodingFilePath, "rb").read())
        data = np.array(data)

        labelIDs = np.unique(labels)

        for labelID in labelIDs:

            logging.info(f"Generating images for Face ID {labelID}")

            FaceFolder = os.path.join(OutputFolderName, f"Face_{labelID}")
            os.makedirs(FaceFolder, exist_ok=True)

            idxs = np.where(labels == labelID)[0]

            portraits = []
            counter = 1

            for i in idxs:
                image = cv2.imread(data[i]["imagePath"])
                (o_top, o_right, o_bottom, o_left) = data[i]["loc"]

                height, width, _ = image.shape

                top = max(o_top - 150, 0)
                bottom = min(o_bottom + 150, height)
                left = max(o_left - 100, 0)
                right = min(o_right + 100, width)

                portrait = image[top:bottom, left:right]

                if len(portraits) < 25:
                    portraits.append(portrait)

                portrait = ResizeUtils().rescale_by_width(portrait, 400)

                cv2.imwrite(
                    os.path.join(FaceFolder, f"face_{counter}.jpg"),
                    portrait
                )

                counter += 1

            if portraits:
                montage = build_montages(portraits, (96, 120), (5, 5))[0]
                cv2.imwrite(
                    os.path.join(MontageFolderPath, f"Face_{labelID}.jpg"),
                    montage
                )
