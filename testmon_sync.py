"""
Script for speeding up testing using testmon.
"""

import sys

from azure.storage.blob import ContainerClient


CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=testmonstorage;AccountKey=TnI/vitXb4j/B1BLdDXLKpgzKMw8XgnKX9CrzaqbmqlVNc4fHjqcy3WUkCTwCd7Hw2fhOnYHvQAP8IvmeQMztQ==;EndpointSuffix=core.windows.net"

if sys.argv[1] == "upload":
    print("uploading .testmondata...")
    with open("tests/.testmondata", "rb") as fp:
        client = ContainerClient.from_connection_string(CONNECTION_STRING, "versions")
        client.upload_blob(f"{sys.argv[2]}-{sys.argv[3]}-{sys.argv[4]}", fp)
elif sys.argv[1] == "download":
    try:
        print(f"downloading .testmondata for commit {sys.argv[2]}-{sys.argv[3]}-{sys.argv[4]}")
        with open("tests/.testmondata", "wb") as fp:
            client = ContainerClient.from_connection_string(CONNECTION_STRING, "versions")
            blob = client.download_blob(f"{sys.argv[2]}-{sys.argv[3]}-{sys.argv[4]}")
            blob.readinto(fp)
    except Exception as e:  # pylint: disable=broad-except
        print(f"could not download .testmondata for commit {sys.argv[2]}-{sys.argv[3]}-{sys.argv[4]}")
else:
    raise ValueError()
