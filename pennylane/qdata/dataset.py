# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains the dataset class.
"""
from base64 import decode
from os import path
import dill
import bz2
import requests
import pickle
import os
import sys
import json


class Dataset:
    "The dataset data type. Allows users to create datasets with their own data"

    def __init__(self, **kwargs):
        self.variables = kwargs


url = "http://127.0.0.1:5001/download"


def load(data_type, data_params, filter_params=None, file_path=None, force=False):
    """Downloads the data if it is not already present in the directory and return it to user as a Datset object"""


def load(directory, datatype, datasubtype, redownload=False):
    """Downloads the data if not already present in directory and
    returns a qml.dataset of the desired dataset represented by
    datatype and datasubtype"""
    # check if the data is in directory
    # if data is not in directory, download data
    # currently using datatype/datasubtype/datasubtype as a place holder
    # for when we have sub and subsubtypes
    dir_path = f"{directory}/{datatype}/{datasubtype}"
    file_path = f"{directory}/{datatype}/{datasubtype}/{datasubtype}"
    if not path.exists(file_path) or redownload:
        if not path.exists(dir_path):
            os.makedirs(dir_path)
        with open(file_path, "wb") as file:
            response = requests.get(
                url + "/" + datatype + "/" + datasubtype, stream=True
            )  # test dataset
            if response.status_code == 200:
                print(f"Downloading data to {file_path}")
                total_length = response.headers.get("content-length")
                if total_length is None:
                    file.write(response.content)
                else:
                    total_length, barsize, progress = int(total_length), 60, 0
                    for idx, chunk in enumerate(response.iter_content(chunk_size=4096)):
                        if chunk:
                            file.write(chunk)
                            progress += len(chunk)
                            completed = min(round(progress / total_length, 3) * 100, 100)
                            barlength = int(progress / total_length * barsize)
                            if not idx % 1000:
                                f = f"[{chr(9608)*barlength} {round(completed, 3)} %{'.'*(barsize-barlength)}] {convert_size(progress)}/{convert_size(total_length)}"
                                sys.stdout.write("\r" + f)
                                sys.stdout.flush()

                    f = f"[{chr(9608)*barlength} {round(completed, 3)} %{'.'*(barsize-barlength)}] {convert_size(progress)}/{convert_size(total_length)}"
                    sys.stdout.write("\r" + f)
                    sys.stdout.flush()
            else:
                response.raise_for_status()

    with open(file_path, "rb") as file:
        try:
            data = dill.load(file)
        except:
            data = pickle.load(file)

    return data


def listdatasets():
    """Returns a list of datasets and their sizes"""
    byteslist = requests.get(url + "/about").content
    data = json.loads(byteslist)

    for key, val in data.items():
        for ky, vl in data[key].items():
            for k, v in data[key][ky].items():
                data[key][ky][k] = convert_size(v)
    return data


def getinfo(datatype, datasubtype, name):
    value = requests.get(url + "/about/" + datatype + "/" + datasubtype + "/" + name).content
    valueasdict = json.loads(value)
    return valueasdict
