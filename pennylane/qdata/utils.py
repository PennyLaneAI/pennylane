import os
import sys
import math
import json
import shutil
import zipfile
import requests
import itertools
from glob import glob

DATA_STRUCT = {
    "qchem": {
        "params": ["molname", "basis", "bondlength"],
        "keys": {},
    },
    "qspin": {
        "params": ["sysname", "periodicity", "lattice", "layout"],
        "keys": {},
    },
}

URL = "http://127.0.0.1:5001"


def _convert_size(size_bytes):
    """Convert file size for the dataset into appropriate units from bytes"""
    if not size_bytes:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    indx = int(math.floor(math.log(size_bytes, 1024)))
    size = round(size_bytes / math.pow(1024, indx), 2)
    return f"{size} {size_name[indx]}"


def _write_prog_bar(progress, completed, barsize, barlength, total_length):
    """Helper function for printing progress bar for downloads"""
    f = f"[{chr(9608)*barlength} {round(completed, 3)} %{'.'*(barsize-barlength)}] {_convert_size(progress)}/{_convert_size(total_length)}"
    sys.stdout.write("\r" + f)
    sys.stdout.flush()


def _validate_params(data_type, data_params, filter_params=None):
    """Validate parameters for loading the data"""

    if data_type not in list(DATA_STRUCT.keys()):
        raise ValueError(
            f"Currently we have data hosted from types: qchem and qspin, but got {data_type}."
        )

    if not isinstance(data_params, dict):
        raise TypeError(f"Args 'data_params' should be a dict, but got {type(data_params)}.")

    if list(data_params.keys()) != DATA_STRUCT[data_type]["params"]:
        raise ValueError(
            f"Supported parameter values for {data_type} are {DATA_STRUCT[data_type]['params']}, but got {list(data_params.keys())}."
        )

    if filter_params is not None and set(filter_params).issubset(DATA_STRUCT[data_type]["keys"]):
        raise ValueError(
            f"Supported key values for {data_type} are {DATA_STRUCT[data_type]['keys']}, but got {filter_params}."
        )

    if filter_params is not None and not isinstance(filter_params, list):
        raise TypeError(f"Args 'filter_params' should be a list, but got {type(filter_params)}.")


def _check_data_exist(data_type, data_params, directory_path):
    exist = False
    if "full" in data_params.values():
        exist = True
    else:
        subdirec_path = [data_params[param] for param in DATA_STRUCT[data_type]["params"]]
        for subdirec in itertools.product(*subdirec_path):
            path = os.path.join(directory_path, *subdirec)
            if not os.path.exists(path) or not glob(
                os.path.join(path, "**", "*.dat"), recursive=True
            ):
                exist = True
                break
    return exist


def _write_data():
    pass


def load(data_type, data_params, filter_params=None, folder_path=None, force=True):
    """Downloads the data if it is not already present in the directory and return it to user as a Datset object"""

    _validate_params(data_type, data_params, filter_params)

    data_params = {
        key: (val if isinstance(val, list) else [val]) for (key, val) in data_params.items()
    }

    directory_path = f"datasets/{data_type}"
    if folder_path is not None:
        if folder_path[-1] == "/":
            folder_path = folder_path[:-1]
        directory_path = f"/{folder_path}/{directory_path}"

    if not force:
        force = _check_data_exist(data_type, data_params, directory_path)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    with open(f"{directory_path}/data.zip", "wb") as file:
        request_data = {
            "dparams": data_params,
            "filters": filter_params if filter_params is not None else ["full"],
        }
        response = requests.post(f"{URL}/download/{data_type}", json=request_data, stream=True)
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
                            _write_prog_bar(progress, completed, barsize, barlength, total_length)
                _write_prog_bar(progress, completed, barsize, barlength, total_length)
        else:
            response.raise_for_status()

    data_files = []
    with zipfile.ZipFile(f"{directory_path}/data.zip", "r") as zpf:
        for file in zpf.namelist():
            if file[-3:] == "pkl":
                data_files.append(file)
        zpf.extractall(f"{directory_path}")
    os.remove(f"{directory_path}/data.zip")

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

    wdata = json.loads(requests.get(URL + "/download/about").content)
    if folder_path is None:
        fdata = None
    else:
        fdata = direc_to_dict(folder_path)
    return wdata, fdata


def dfs(t, path=[]):
    """Perform directory structure DFS"""
    if isinstance(t, dict):
        for key, val in t.items():
            yield from dfs(val, [*path, key])
    else:
        yield path, t


def get_params(data, data_type, **kwargs):
    """Prepare data_param using listdatasets result"""
    params = DATA_STRUCT[data_type]["params"]
    if not set(kwargs.keys()).issubset(params):
        raise ValueError(
            f"Expected kwargs for the module {module} are {params}, but got {list(kwargs.items())}"
        )

    data_params = [["full"] for params in params]
    mtch_params = []
    for key, val in kwargs.items():
        data_params[params.index(key)] = val if isinstance(val, list) else [val]
        mtch_params.append(params.index(key))

    traverse_data = list(
        filter(
            lambda x: all(
                [
                    x[0][m] in data_params[m]
                    if m < len(params) - 1
                    else set(data_params[m]).issubset(x[1])
                    for m in mtch_params
                ]
            ),
            dfs(data[data_type], []),
        )
    )

    data_params = []
    for data in traverse_data:
        dparams = {param: ["full"] for param in params}
        for idx in mtch_params:
            dparams[params[idx]] = [data[0][idx]] if idx < len(params) - 1 else data[1]
        data_params.append(dparams)
        if not mtch_params:
            break

    return data_params


def getinfo(datatype, datasubtype, name):
    value = requests.get(URL + "/about/" + datatype + "/" + datasubtype + "/" + name).content
    valueasdict = json.loads(value)
    return valueasdict
