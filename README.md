# The GeWaPro package/repository
GeWaPro (Germanium-detector Waveform Processor / GELINA Waveform Processor) is a git repository which combines and compiles methods to analyse the waveforms that are produced by the HPGe detectors at the GAINS setup in the JRC Geel. It allows for easy training of various AI-models on the waveform output data produced at said facility. The end-goal of this machine learning analysis is time resolution improvement of the HPGe detectors.

### Quick-start guide
1. Clone the contents of the [*GeWaPro* git repository](https://github.com/Th0masDam/GeTimeResImprovement) into a local *GeWaPro* folder (use the green ``<> Code`` button on the webpage to either dowload the folder via the command line with ``HTTPS``, or download as a zipped folder and extract its contents).
2. Navigate to the local *GeWaPro* folder and create a folder inside it called ``.venv``.
3. Download and install [Python 3.11.0 or higher](https://www.python.org/downloads/), which comes with the PIP installer automatically.
4. Within a terminal run ``pip install poetry`` to install package manager *Poetry*.
5. Run from your local *GeWaPro* folder: ``poetry install``. This will install all required packages into a local virtual environment within the ``.venv`` folder.
    - You can check if the installation has succeeded with ``poetry env info --path``. This should return the path of the ``.venv`` folder you just created in step 4. Otherwise, remove the folder that the command points to and retry step 2, then retry step 5.
6. The Jupyter notebook file ``waveform_fitting.ipynb`` can now be opened in your browser with command ``poetry run jupyter notebook`` (run from *GeWaPro* folder) or in your preferred IDE/coding environment such as *VSCode*.
    - If not opening the notebooks with the above command, make sure the selected kernel is the one in the ``.venv`` folder. Otherwise, the required packages may not be present and the notebook will not run.

### Usage tutorial
For a tutorial on how to use *GeWaPro*, see the [Tutorial.ipynb notebook](https://github.com/Th0masDam/GeTimeResImprovement/blob/main/Tutorial.ipynb).

### Bugs and questions
For bug reporting and questions regarding GeWaPro, use the [standard issue reporting in GitHub](https://github.com/Th0masDam/GeTimeResImprovement/issues) or contact the maintainer [Th0masDam](https://github.com/Th0masDam).