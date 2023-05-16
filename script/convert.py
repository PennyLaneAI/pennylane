from pennylane.data_old import load
from pennylane.data import Dataset
from pennylane.data.base.attribute import match_obj_type

import zarr


def convert_dataset(ds_old, zroot: zarr.Group):
    ds_new = Dataset(bind=(zroot, ds_old._description))

    for attr in ds_old.attrs:
        print(f"Converting attribute {attr}")
        setattr(ds_new, attr, getattr(ds_old, attr))
        print(f"Converted attribute {attr}")

        delattr(ds_old, attr)



def main():
    ds_old = load('qchem', molname="CH4", bondlength="0.5", basis="STO-3G")[0]

    with zarr.group() as zroot:
        convert_dataset(ds_old, zroot)

        path = f"datasts/zarr/qchem/{ds_old._description}"
        print(f"saving to {path}")

        with zarr.open_group(path, mode="w") as fgrp:
            zarr.copy_all(zroot, fgrp)
    




#ds = Dataset()
#attrs = list(ds_old.attrs)
#shuffle(attrs)


if __name__ == '__main__':
    main()


#for attr_name in attrs:
#    attr = getattr(ds_old, attr_name)
#    print(f"Setting attribute {attr_name}, type={type(attr)}")
#    setattr(ds, attr_name, attr)

