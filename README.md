# Haptic Knowledge Transfer Between Heterogeneous Robots using Kernel Manifold Alignment

<img src="figs/framework.jpg" alt="drawing" width="600px"/>

## Development Environment

`Python 3.7.6` and `MATLAB R2019b update 3 (9.7.0.1261785)` are used for development and following packages are required to run the code:<br><br>

### Python Dependencies
`pip install scipy==1.1.0`<br>
`pip install sklearn=0.22.1`<br>
`pip install numpy==1.18.1`<br>
`pip install tensorflow-gpu==2.0.0`<br>
`pip install matplotlib==3.1.2`<br>
`MATLAB Engine API for Python`

### MATLAB Dependencies
`Statistics and Machine Learning Toolbox`

## [Dataset](Datasets)

- [Visualization of each modalities](DatasetVisualization.ipynb)

### Dataset Collection

Baxter: https://github.com/medegw01/baxter_tufts <br>
Fetch: https://github.com/gtatiya/Fetch_Pick_and_Place <br>
Sawyer: https://github.com/medegw01/sawyer_tufts

## How to run the code?

### Speeding up object recognition

Baseline condition: `python MulitBehaviorsObjectClassification.py` <br>
Transfer condition: `python MulitBehaviorsOC_KEMA.py`

### Novel object recognition

`python MulitBehaviorsNovelOC_KEMA.py`
