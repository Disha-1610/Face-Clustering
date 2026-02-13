import os
import shutil
import time
import uuid
import logging
from FaceClusteringLibrary import *

# ---------------- Logging ----------------
logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":

    # -------- Job Based Folder --------
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    base_job_path = os.path.join("jobs", job_id)
    os.makedirs(base_job_path, exist_ok=True)

    logging.info(f"Starting job {job_id}")

    video_path = "Footage.mp4"
    frames_dir = os.path.join(base_job_path, "Frames")
    encodings_dir = os.path.join(base_job_path, "Encodings")
    clustered_dir = os.path.join(base_job_path, "ClusteredFaces")

    # ---------- Generate Frames ----------
    try:
        framesGenerator = FramesGenerator(video_path)
        framesGenerator.GenerateFrames(frames_dir, frame_interval=10)
        logging.info("Frames extracted successfully")
    except Exception as e:
        logging.error(f"Frame extraction failed: {e}")
        raise

    # ---------- Encoding Pipeline ----------
    os.makedirs(encodings_dir, exist_ok=True)

    pipeline = Pipeline(
        FramesProvider("Files source", sourcePath=frames_dir) |
        FaceEncoder("Encode faces") |
        DatastoreManager("Store encoding", encodingsOutputPath=encodings_dir),
        n_threads=1,
        quiet=True
    )

    pbar = TqdmUpdate()
    pipeline.run(update_callback=pbar.update)

    logging.info("Encodings extracted")

    # ---------- Merge Pickles ----------
    encoding_pickle_path = os.path.join(base_job_path, "encodings.pickle")

    picklesListCollator = PicklesListCollator(encodings_dir)
    picklesListCollator.GeneratePickle(encoding_pickle_path)

    time.sleep(0.3)

    # ---------- Clustering ----------
    try:
        faceClusterUtility = FaceClusterUtility(encoding_pickle_path)
        faceImageGenerator = FaceImageGenerator(encoding_pickle_path)

        labelIDs = faceClusterUtility.Cluster()
        faceImageGenerator.GenerateImages(labelIDs, clustered_dir, "Montage")

        logging.info("Clustering completed successfully")
    except Exception as e:
        logging.error(f"Clustering failed: {e}")
        raise

    logging.info(f"Job {job_id} completed")


