from google.cloud import storage
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cs585-final-project-b611a901a4b2.json"

BUCKET_NAME = os.getenv("BUCKET_NAME", "dataproc-staging-us-central1-51800608968-pazlrfhf")
BUCKET_PATH = os.getenv("BUCKET_PATH", "data/")

print(f"BUCKET_NAME={BUCKET_NAME} BUCKET_PATH={BUCKET_PATH}")
print(f"via CLI: gsutil ls -l gs://{BUCKET_NAME}/{BUCKET_PATH}/")

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
# content_list = list(bucket.list_blobs(prefix=f"{BUCKET_PATH}/"))
# print(content_list)
blob = bucket.get_blob('data/testing.txt')
print(blob.download_as_string())

elements = bucket.list_blobs()
files=[a.name for a in elements]
print(files)