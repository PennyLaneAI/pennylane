from pennylane.data_old import load
from pennylane.data import Dataset
from pennylane.data.attributes import QChemHamiltonian
import h5py
from pennylane import Hamiltonian

def convert_dataset(ds_old, zroot: h5py.File):
    ds_new = Dataset(bind=(zroot, ds_old._description))

    for attr_name in ds_old.attrs:
        print(f"Converting attribute {attr_name}")
        attr = getattr(ds_old, attr_name)

        if isinstance(attr, Hamiltonian):
            print("using QCHemHamiltonaian")
            attr = QChemHamiltonian(attr, parent_and_key=(ds_new.bind, attr_name))
        else:
            setattr(ds_new, attr_name, attr)
        print(f"Converted attribute {attr_name}")

        delattr(ds_old, attr_name)



def main():
    ds_old = load('qchem', molname="CH4", bondlength="0.5", basis="STO-3G")[0]

    with h5py.File(f"datasets/test.h5py", "w-") as zroot:
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

