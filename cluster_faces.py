
from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import os
import pickle
import cv2
import shutil
import time
import logging

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



if __name__ == "__main__":
	EncodingPickleFilePath = "encodings.pickle"

	faceClusterUtility = FaceClusterUtility(EncodingPickleFilePath)
	faceImageGenerator = FaceImageGenerator(EncodingPickleFilePath)

	labelIDs = faceClusterUtility.Cluster()
	faceImageGenerator.GenerateImages(labelIDs, "ClusteredFaces", "Montage")

