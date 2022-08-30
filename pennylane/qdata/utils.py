import os
import sys
import math
import json
import shutil
import requests
import itertools
from glob import glob

DATA_STRUCT = {
    "qchem": {
        "params": ["molname", "basis", "bondlength"],
        "keys": {},
    },
    "qspin": {
        "params": ["sysname", "periodicity", "lattice"],
        "keys": {},
    },
}

URL = "http://127.0.0.1:5001"


def convert_size(size_bytes):
    """Convert file size for the dataset into appropriate units from bytes"""
    if not size_bytes:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    indx = int(math.floor(math.log(size_bytes, 1024)))
    size = round(size_bytes / math.pow(1024, indx), 2)
    return f"{size} {size_name[indx]}"


def write_prog_bar(progress, completed, barsize, barlength, total_length):
    """Helper function for printing progress bar for downloads"""
    f = f"[{chr(9608)*barlength} {round(completed, 3)} %{'.'*(barsize-barlength)}] {convert_size(progress)}/{convert_size(total_length)}"
    sys.stdout.write("\r" + f)
    sys.stdout.flush()


def load(data_type, data_params, filter_params=None, folder_path=None, force=True):
    """Downloads the data if it is not already present in the directory and return it to user as a Datset object"""

    if data_type not in list(DATA_STRUCT.keys()):
        raise ValueError(
            f"Currently we have data hosted from types: qchem and qspin, but got {data_type}."
        )

    if list(data_params.keys()) != DATA_STRUCT[data_type]["params"]:
        raise ValueError(
            f"Supported parameter values for {data_type} are {DATA_STRUCT[data_type]['params']}, but got {list(data_params.keys())}."
        )

    if filter_params is not None and set(filter_params).issubset(DATA_STRUCT[data_type]["keys"]):
        raise ValueError(
            f"Supported key values for {data_type} are {DATA_STRUCT[data_type]['keys']}, but got {filter_params}."
        )

    data_params = {
        key: (val if isinstance(list, val) else [val]) for (key, val) in data_params.items()
    }

    directory_path = f"datasets/{data_type}/"
    if folder_path is not None:
        if folder_path[-1] == "/":
            folder_path = folder_path[:-1]
        directory_path = f"{folder_path}/{directory_path}"

    if not force:
        if "full" in data_params.values():
            force = True
        else:
            subdirec_path = [data_params[param] for param in DATA_STRUCT[data_type]["params"]]
            for subdirec in itertools.product(*subdirec_path):
                path = os.path.join(directory_path, *subdirec)
                if not os.path.exists(path) or not glob(
                    os.path.join(path, "**", "*.dat"), recursive=True
                ):
                    force = True
                    break

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    with open(f"{directory_path}/data.zip", "wb") as file:
        request_data = {
            "datatype": data_type,
            "parameters": data_params,
            "filters": filter_params,
        }
        response = requests.post(f"{URL}/download", data=request_data, stream=True)
        if response.status_code == 200:
            print(f"Downloading data to {directory_path}")
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
                            write_prog_bar(progress, completed, barsize, barlength, total_length)
                write_prog_bar(progress, completed, barsize, barlength, total_length)
        else:
            response.raise_for_status()

    shutil.unpack_archive(f"{directory_path}/data.zip")
    data_files = []
    for file in sorted(glob(os.path.join(directory_path, "**", "*.dat"), recursive=True)):
        data_files.append(file)

    return data_files


def direc_to_dict(path):
    """Helper function to create dictionary structure from directory path"""
    for root, dirs, files in os.walk(path):
        if dirs:
            tree = {x: direc_to_dict(os.path.join(root, x)) for x in dirs}
            vals = [x is None for x in tree.values()]
            if all(vals):
                return list(dirs)
            if any(vals):
                for key, val in tree.items():
                    if val is None:
                        tree.update({key: []})
            return tree
        return None


def listdatasets(folder_path=None):
    """Returns a list of datasets and their sizes"""

    wdata = json.loads(requests.get(url + "/about").content)
    if folder_path is None:
        fdata = None
    else:
        fdata = direc_to_dict(folder_path)
    return wdata, fdata


def getinfo(datatype, datasubtype, name):
    value = requests.get(url + "/about/" + datatype + "/" + datasubtype + "/" + name).content
    valueasdict = json.loads(value)
    return valueasdict
