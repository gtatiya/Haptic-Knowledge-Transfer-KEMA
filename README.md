# Haptic Knowledge Transfer Between Heterogeneous Robots using Kernel Manifold Alignment

**Abstract:**

> Humans learn about object properties using multiple modes of perception. Recent advances show that robots can use non-visual sensory modalities (i.e., haptic and tactile sensory data) coupled with exploratory behaviors (i.e., grasping, lifting, pushing, dropping, etc.) for learning objects' properties such as shape, weight, material and affordances. However, non-visual sensory representations cannot be easily transferred from one robot to another, as different robots have different bodies and sensors. Therefore, each robot needs to learn its task-specific sensory models from scratch. To address this challenge, we propose a framework for knowledge transfer using kernel manifold alignment (KEMA) that enables source robots to transfer haptic knowledge about objects to a target robot. The idea behind our approach is to learn a common latent space from multiple robots' feature spaces produced by respective sensory data while interacting with objects. To test the method, we used a dataset in which 3 simulated robots interacted with 25 objects and showed that our framework speeds up haptic object recognition and allows novel object recognition.

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

### Discretized Mean 10 bins

<img src="figs/Effort_Data_discretizedmean-10_GnBu.jpg" alt="drawing" width="900px"/>

### Discretized Range 15 bins

<img src="figs/Effort_Data_discretizedrange-15_GnBu.jpg" alt="drawing" width="900px"/>

### Dataset Collection

Baxter: https://github.com/medegw01/baxter_tufts <br>
Fetch: https://github.com/gtatiya/Fetch_Pick_and_Place <br>
Sawyer: https://github.com/medegw01/sawyer_tufts <br>

<table>

<tr>
<td>
	<a href="https://github.com/medegw01/baxter_tufts">Baxter</a>
	<img src="figs/Baxter_pick_and_place_shake.gif" alt="drawing" width="800" height="250"/>
</td>

<td>
	<a href="https://github.com/gtatiya/Fetch_Pick_and_Place">Fetch</a>
	<img src="figs/Fetch_grasp_pick_hold_shake_place.gif" alt="drawing" width="800" height="250"/>
</td>

<td>
	<a href="https://github.com/medegw01/sawyer_tufts">Sawyer</a>
	<img src="figs/Sawyer_pick_and_place_shake.gif" alt="drawing" width="800" height="250"/>
</td>
</tr>

</table>

## How to run the code?

### Speeding up object recognition

Baseline condition: `python MulitBehaviorsObjectClassification.py` <br>
Transfer condition: `python MulitBehaviorsOC_KEMA.py`

### Novel object recognition

`python MulitBehaviorsNovelOC_KEMA.py`

## Results (Discretized Mean 10 bins)

### Illustrative Example

<img src="Results/IllustrativeExample.jpg" alt="drawing" width="900px"/>

### Speeding up object recognition results

#### Baster as Target Robot

<img src="Results/SpeedingUp_Baxter.jpg" alt="drawing" width="900px"/>

#### Fetch as Target Robot

<img src="Results/SpeedingUp_Fetch.jpg" alt="drawing" width="900px"/>

#### Sawyer as Target Robot

<img src="Results/SpeedingUp_Sawyer.jpg" alt="drawing" width="900px"/>

### Novel object recognition results

#### Baster as Target Robot

<img src="Results/NovelObj_Baxter.jpg" alt="drawing" width="900px"/>

#### Fetch as Target Robot

<img src="Results/NovelObj_Fetch.jpg" alt="drawing" width="900px"/>

#### Sawyer as Target Robot

<img src="Results/NovelObj_Sawyer.jpg" alt="drawing" width="900px"/>
