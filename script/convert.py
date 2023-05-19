from pennylane.data_old import load
from pennylane.data.qchem import QChemDataset
import h5py

rename_map = {"meas_groupings": "qwc_groupings"}
skiplist = {"parameters"}

def convert_dataset(ds_old, zroot: h5py.File):

    ds_new = QChemDataset(zroot, validate=False)

    for attr_name in ds_new.fields:
        if attr_name in skiplist:
            continue
        
        print(f"Converting attribute {attr_name}")
        attr = getattr(ds_old, rename_map.get(attr_name, attr_name))
        setattr(ds_new, attr_name, attr)
        print(f"Converted attribute {attr_name}")

        delattr(ds_old, attr_name)



def main():
    ds_old = load('qchem', molname="CH4", bondlength="0.5", basis="STO-3G")[0]
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

