from pennylane.data_old import load
from pennylane.data import Dataset
import h5py

def convert_dataset(ds_old, zroot: h5py.File):

    ds_new = Dataset((zroot, "data"))

    for attr_name in ds_old.attrs:
        print(f"Converting attribute {attr_name}")
        attr = getattr(ds_old, attr_name)
        setattr(ds_new, attr_name, attr)
        print(f"Converted attribute {attr_name}")

        delattr(ds_old, attr_name)



def main():
    ds_old = load('qchem', attributes=["qwc_groupings"], molname="CH4", bondlength="0.5", basis="STO-3G")[0]
    with h5py.File(f"datasets/test.h5", "w-") as zroot:
        convert_dataset(ds_old, zroot)

    




#ds = Dataset()
#attrs = list(ds_old.attrs)
#shuffle(attrs)


if __name__ == '__main__':
    main()


#for attr_name in attrs:
#    attr = getattr(ds_old, attr_name)
#    print(f"Setting attribute {attr_name}, type={type(attr)}")
#    setattr(ds, attr_name, attr)

