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
import json

class Dataset:
    "The dataset data type. Allows users to create datasets with their own data"
    def __init__(self, **kwargs):
        self.variables = kwargs

def load(directory,datatype,datasubtype, redownload=False):
    """Downloads the data if not already present in directory and 
    returns a qml.dataset of the desired dataset represented by 
    datatype and datasubtype"""
    #check if the data is in directory
    #if data is not in directory, download data
    #currently using datatype/datasubtype/datasubtype as a place holder
    #for when we have sub and subsubtypes
    if not path.exists(directory+'/'+datatype+'/'+datasubtype+'/'+datasubtype) or redownload:
        print("downloading data")
        if not path.exists(directory+'/'+datatype+'/'+datasubtype):
            os.makedirs(directory+'/'+datatype+'/'+datasubtype)
        with open(directory+'/'+datatype+'/'+datasubtype+'/'+datasubtype, 'wb') as file:
            response = requests.get('http://127.0.0.1:5000/download/'+datatype+'/'+datasubtype) #test dataset
            dataset = response.content
            file.write(dataset)
    
    with open(directory+'/'+datatype+'/'+datasubtype+'/'+datasubtype, 'rb') as file:
        data = pickle.load(file)

    return data

def listdatasets():
    """Returns a list of datasets and their sizes"""
    byteslist = requests.get('http://127.0.0.1:5000/download/about').content
    return json.loads(byteslist)

def getinfo(datatype,datasubtype,name):
    value = requests.get('http://127.0.0.1:5000/download/about/'+datatype+'/'+datasubtype+'/'+name).content
    valueasdict = json.loads(value)
    return valueasdict