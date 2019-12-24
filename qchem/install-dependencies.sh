# install openbabel
sudo apt-get -qq update
sudo apt-get install -y openbabel

# install miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
source "$HOME/miniconda/etc/profile.d/conda.sh"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda

# Useful for debugging any issues with conda
conda info -a

# create a virtual env
conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION pytest pytest-cov
conda activate test-environment

# Install psi4
conda install psi4 psi4-rt python=$TRAVIS_PYTHON_VERSION -c psi4
# conda install -c conda-forge openbabel
# ln -s $HOME/miniconda/envs/test-environment/bin/obabel $HOME/miniconda/envs/test-environment/bin/babel

# install PennyLane and Qchem
pip install -r requirements.txt
python setup.py bdist_wheel
pip install dist/PennyLane*.whl
cd qchem && python setup.py bdist_wheel && cd ../
pip install qchem/dist/PennyLane_Qchem*.whl
