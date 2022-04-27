import sys

import git
from azure.storage.blob import ContainerClient


CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=testmonstorage;AccountKey=TnI/vitXb4j/B1BLdDXLKpgzKMw8XgnKX9CrzaqbmqlVNc4fHjqcy3WUkCTwCd7Hw2fhOnYHvQAP8IvmeQMztQ==;EndpointSuffix=core.windows.net"

if len(sys.argv) == 1:
    print("uploading .testmondata...")
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    with open(".testmondata", "rb") as fp:
        client = ContainerClient.from_connection_string(CONNECTION_STRING, "versions")
        client.upload_blob(sha, fp)
else:
    commit_hash = sys.argv[1]
    print(f"downloading .testmondata for commit {commit_hash}")
    with open(".testmondata", "wb") as fp:
        client = ContainerClient.from_connection_string(CONNECTION_STRING, "versions")
        blob = client.download_blob(commit_hash)
        blob.readinto(fp)
