"""
Script for speeding up testing using testmon.
"""

import sys
import os

from azure.storage.blob import ContainerClient


CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=testmonstorage;AccountKey=TnI/vitXb4j/B1BLdDXLKpgzKMw8XgnKX9CrzaqbmqlVNc4fHjqcy3WUkCTwCd7Hw2fhOnYHvQAP8IvmeQMztQ==;EndpointSuffix=core.windows.net"

if sys.argv[1] == "upload":
    print("uploading .testmondata...")
    with open(".testmondata", "rb") as fp:
        client = ContainerClient.from_connection_string(CONNECTION_STRING, "versions")
        client.upload_blob('-'.join(sys.argv[2:]), fp)
elif sys.argv[1] == "download":
    try:
        print(f"downloading .testmondata for commit {'-'.join(sys.argv[2:])}")
        client = ContainerClient.from_connection_string(CONNECTION_STRING, "versions")
        blob = client.download_blob('-'.join(sys.argv[2:]))
        print(f"saving to {os.getcwd()}/.testmondata")
        with open(".testmondata", "wb") as fp:
            blob.readinto(fp)
    except Exception as e:  # pylint: disable=broad-except
        print(f"could not download .testmondata for commit {'-'.join(sys.argv[2:])}")
else:
    raise ValueError()
