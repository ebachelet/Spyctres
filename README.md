# Spyctres

Spyctres is a tool for analysing stellar spectra.  It can compare a measured spectrum with publicly-available 
libraries of spectral templates in order to find the closest match.  

Developer: Etienne Bachelet

Please note that Spyctres is still under development.  

## Installation
For the time being, the best way to install Spyctres is to clone its [Github repository](https://github.com/ebachelet/Spyctres), 
and run the setup.py.  It is recommended that you create and activate a virtual environment before proceeding with this
installation. 

```commandline
venv> git clone https://github.com/ebachelet/Spyctres.git
venv> cd Spyctres/
venv> python setpy.py install
```

Spyctres makes use of ```pysynphot``` and its successor package ```stsynphot``` as dependencies.  
These can both be installed via pip:

```commandline
venv> pip install pysynphot
venv> pip install stsynphot
```

Lastly, you will need to download the stellar spectrum library used as templates in Spyctres fitting 
process.  Tarballs of these libraries are linked from the [```pysynphot``` ReadTheDocs page](https://pysynphot.readthedocs.io/en/latest/index.html#pysynphot-installation-setup)
 in the second table.

Download and unpack these files into a local directory.  In order to tell Spyctres where to find this library, 
the path to it should be declared as an environment variable in any script or shell where the software is 
run, e.g. 

```python
os.environ['PYSYN_CDBS'] =  '/my/path/cdbs/'
```

## Running Spyctres
The ```quick_example.py``` script in this repository provides a worked demonstration of the fitting process. 