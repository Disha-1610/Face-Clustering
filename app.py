import os
import uuid
import logging
from flask import Flask, request, jsonify

import cloudinary
import cloudinary.uploader

from FaceClusteringLibrary import *

# ---------------- Flask App ----------------
app = Flask(__name__)

# ---------------- Logging ----------------
logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Folder to store uploads + jobs
UPLOAD_FOLDER = "uploads"
JOBS_FOLDER = "jobs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(JOBS_FOLDER, exist_ok=True)

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)
# ---------------- Home Route ----------------
@app.route("/")
def home():
    return "✅ Face Clustering API is Running!"


# ---------------- Cluster Route ----------------
@app.route("/cluster", methods=["POST"])
def cluster_faces():
    """
    Upload a video and run face clustering.
    """

    # 1. Check if video file exists in request
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files["video"]

    if video_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # 2. Save uploaded video
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job_path = os.path.join(JOBS_FOLDER, job_id)

    os.makedirs(job_path, exist_ok=True)

    video_path = os.path.join(job_path, video_file.filename)
    video_file.save(video_path)

    logging.info(f"Starting clustering job {job_id}")

    # Output folders
    frames_dir = os.path.join(job_path, "Frames")
    encodings_dir = os.path.join(job_path, "Encodings")
    clustered_dir = os.path.join(job_path, "ClusteredFaces")

    try:
        # ---------- Generate Frames ----------
        framesGenerator = FramesGenerator(video_path)
        framesGenerator.GenerateFrames(frames_dir, frame_interval=1)

        # ---------- Encoding Pipeline ----------
        os.makedirs(encodings_dir, exist_ok=True)

        pipeline = Pipeline(
            FramesProvider("Files source", sourcePath=frames_dir)
            | FaceEncoder("Encode faces")
            | DatastoreManager("Store encoding", encodingsOutputPath=encodings_dir),
            n_threads=3,
            quiet=True
        )

        pbar = TqdmUpdate()
        pipeline.run(update_callback=pbar.update)

        # ---------- Merge Pickles ----------
        encoding_pickle_path = os.path.join(job_path, "encodings.pickle")

        picklesListCollator = PicklesListCollator(encodings_dir)
        picklesListCollator.GeneratePickle(encoding_pickle_path)

        # ---------- Clustering ----------
        faceClusterUtility = FaceClusterUtility(encoding_pickle_path)
        faceImageGenerator = FaceImageGenerator(encoding_pickle_path)

        labelIDs = faceClusterUtility.Cluster()
        faceImageGenerator.GenerateImages(labelIDs, clustered_dir, "Montage")

        logging.info(f"Job {job_id} completed successfully")

        # 3. Return response
        return jsonify({
            "message": "Clustering completed!",
            "job_id": job_id,
            "output_folder": clustered_dir
        })

    except Exception as e:
        logging.error(f"Clustering failed: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------- Run Server ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)




