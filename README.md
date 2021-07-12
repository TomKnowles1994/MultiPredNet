# MultiPredNet Experiment

## Overview

### The Algorithm

MultiPredNet is a Predictive Coding-inspired algorithm designed to work with multimodal data, in this case visual and tactile. The network continually attempts to predict the incoming sensory information by passing predictions to downstream layers, with the resulting error from these predictions used to adjust weights. Each predictive layer has its own representation space that is continually refined from these error updates, with most of the network devoted to predicting either visual or tactile data. The topmost layer attempts to predict the incoming activity of both visual *and* tactile streams, and thus builds a top-level multimodal representation. This builds upon \[1\] and \[2\]. All further references to MultiPredNet's representation space refers to this top-level, multimodal representation, not those of the lower layers.

### The Robot

To gather suitable visual and tactile data, the rat-inspired WhiskEye robot is set up to explore an arena filled with simple 3D shapes, such as coloured cylinders and cubes. Equipped with both camera 'eyes' and arrays of motile whiskers sensitive to deflection, it can capture both visual and tactile sensory information. The robot uses a tactile saliency map to explore novel and interesting areas, and a central pattern generator combined with local, per-whisker controllers provides bursts of whisking to gather tactile impressions. WhiskEye's cameras take a visual snapshot of the scene at the moment of maximal whisker protraction, synchronising the tactile and visual inputs. This ensures that both visual and tactile inputs are taken from the same timestep, to better allow for learning of multi-sensory representations. Further details can be found in \[3\].

### The Rationale

For this experiment, the representation space learned by MultiPredNet is intended to be useful for place recognition; that is, samples from representation space should vary according to changes in the robot's sensory input that indicate a change in location or orientation. For example, two representation samples should be significantly seperated from each other if the view of the robot flips 180 degrees, and yet further still if it is also on the other side of the room.

It is worth noting that MultiPredNet has no understanding of the wider context of the experiment. It is a disembodied algorithm with no knowledge of the WhiskEye robot, the methods of data collection, the objects placed in the arena, or the pose of the robot at any given time; all it receives are snapshots of visual and tactile data. This requires it to build a suitable representation of its environment from data alone, learning to predict incoming visuotactile impressions from the robot during its exploration will organically build an unambiguous representation of the environment.

### The Results

The final output of the experiment is a Representational Dissimilarity Matrix (RDM). An RDM is created by measuring the distance between each sample from every other within its native space. These distance metrics are formed into a matrix, visualised as a colourmap. This provides an illustration of the dissimilarity between all samples within a given representation space; regardless of dimensions, modality or application-specific parameters, the distances between each sample to every other provides a 2D matrix of positive scalar values. This enables representations from vastly different spaces and experiments to be compared to one-another. Further information on the Representational Similarity Analysis approach can be found in \[4\].

### The Analysis

To judge whether this learned RDM is useful, a seperate RDM is constructed. This takes data from the robot's pose, also measured from the simulator, and carries out the same process, measuring the distance between samples in Pose Space. Pose Space, given by the comparatively minimal representation of (x, y, theta), is *a priori* known to be a good, unambiguous representation for place recognition; if a robot's coordinates in Pose Space are known, then its (x, y) location in the arena and (theta) orientation are also known. This therefore gives a quantifiable quality standard; if the RSM of a given representation correlates well with Pose Space's RDM, then it is useful for place recognition to the degree to which it correlates.

Comparing this entails analysing the covariance of the two RDMs using Spearman's Rank Correlation coefficient. A high coefficient shows that the RDMs, and therefore the representation spaces, co-vary to a high degree. As seen in the example results below, it can be inferred that MultiPredNet, without any notion of pose space in its learning algorithm or information about the world aside for its sensors, has nonetheless learned a representation that varies in a similar way to Pose Space, and therefore is useful for place recognition.

### References

\[1\]  S. Dora, C. Pennartz, S. Bohte, A  Deep  Predictive  Coding  Network  for  Inferring  Hierarchical  Causes  Underlying  Sensory  Inputs,  International  Conference  on  Artificial  NeuralNetworks,  Springer,  Cham,  2018. 
Available: https://pure.ulster.ac.uk/ws/files/77756390/ICANN_submission.pdf

\[2\] O.  Struckmeier,  K.  Tiwari,  S.  Dora,  M.  J.  Pearson,  S.  M.  Bohte,  C.  M.  Pennartz,  and  V.Kyrki, Mupnet:  Multi-modal  predictive  coding  network  for  place  recognition  by  unsupervised learning of joint visuo-tactile latent representations, 2019. arXiv:1909.07201

\[3\] O. Struckmeier, K. Tiwari, M. Pearson, and V. Kyrki, “Vita-slam: A bio-inspired visuo-tactileslam for navigation while interacting with aliased environments,” Jun. 2019.

\[4\] N Kriegeskorte, M Mur, and P. Bandettini. Front. Syst. Neurosci., 24 November 2008 | https://doi.org/10.3389/neuro.06.004.2008

# Replication

## Dependencies:

- Python 3.5 or higher
- Tensorflow 2.0 or higher
- MATLAB 2020b with valid License
- MATLAB Add-ons installed: Robotic Systems Toolbox and Signal Processing Toolbox
- ROS (any version)

## Data:

Suitable data sources can be any of the following:-

- A ROSbag file, generated from the NeuroRobotics Platform Whiskeye Experiment. Instructions for extraction to follow.
- The TAROS 2021 datasets, available at: https://we.tl/t-wPpFnK2TS6
- Alternatively, sample ROSbag and MATLAB files are available at http://doi.org/10.25493/TSTK-AKK

NB: If your data has already been unpacked from its ROSbag, then neither ROS nor the ROSbag is required. Use the generated .mat files instead.

## Installing Dependencies:

- Visit https://www.python.org/ to download Python. Install it.
- Open Terminal and run `pip install tensorflow`
- Visit https://www.mathworks.com/ to download MATLAB 2020b. Install it.
- MATLAB will prompt for a license; enter your organisation credentials to obtain the license. MATLAB will find and activate automatically.

## Recommended Folder Structure

There are many output files required for this experiment to run successfully, and many different file paths required. From experience, it is very easy to accidentally overwrite output files without an appropriate folder structure set up ahead of time. It is recommended to have a seperate folder for this experiment (likely the one this repository has been downloaded or cloned to), with two subfolder trees; one to store data downloaded from the link about, and one for storing script outputs.

Before any data is extracted or downloaded, the folder structure should look something like:
```                   
MultiPredNet  
└───datasets
│   └───trainingset
│   └───testset1
│   └───testset2
│   └───testset3
│   └───testset4
└───representations
    └───testset1
            └───both
            └───visual
            └───tactile
    └───testset2
            └───both
            └───visual
            └───tactile
    └───testset3
            └───both
            └───visual
            └───tactile
    └───testset4
            └───both
            └───visual
            └───tactile
```

The easiest way to create this structure is to run `build_multiprednet_folder_structure.py` in the directory you wish to store your MultiPredNet results in. This will create all the necessary folders, and can be used to regenerate the missing parts of an existing folder structure.

## Configuration Steps:

After cloning this repository:

1) Extract data from ROSbag OR download sample .mat data from the link provided
2) Alter the three Python scripts by substituting the placeholder paths with your own
3) Run the scripts

## 1) ROSbag extraction (if required)

Within Terminal:

1) Build a catkin workspace for this experiment
2) Copy the `whiskeye_plugin` folder into the `/home/$USER/catkin_ws/src/matlab_msgs/` folder
3) Run `catkin_make` in the top level of your catkin workspace (`catkin_ws`)

Within Terminal:

4) `cd` into the whisker_capture folder.
5) Run `roscore`. ROS startup will commence.

Within MATLAB:

5) Run `rosinit`. MATLAB will create a Python virtual environment and initialise a ROS node.
6) Run `rosgenmsg('/home/$USER/catkin_ws/src/matlab_msgs/')`. The custom message types will be registered and the build process started.
7) Follow the instructions provided in the MATLAB Command Window. If `savepath` gives an error, you may need to `chown $USER` the `pathdef.m` file mentioned.

Within a 2nd Terminal:

8) `cd` to the directory containing your .bag file. Usually this will be named '1.bag'.
9) Run `rosbag play 1.bag`. The console will begin running through the ROSbag history.

Finally, within MATLAB:

10) Run `import_rosbag.m`. The script will find the revevant topics and output the corresponding MATLAB files.

## 2) Python Script Alterations

### MultiPredNet Training Script

Within `python_multiprednet_train_showcase.py`:

- [Line 61]: assign your training .mat data file location to the `data_path` variable. The folder must already exist.
- [Line 63]: assign a suitable save path to the `save_path` variable; it is recommended to create a new, empty folder for this purpose
- [Line 64]: assign a suitable load path to the `load_path` variable; this can be the same as your `save_path`

### MultiPredNet Inference Script

Within `python_multiprednet_gen_reps_showcase.py`:

- [Line 14]: assign trained model location to the `model_path` variable. This is like to be the same as `save_path` within the Training Script
- [Line 15]: assign training data location to the `tr_data_path` variable
- [Line 16]: assign test data location to the `ts_data_path` variable
- [Line 17]: assign output location for representations to the `save_path` variable. This can be the same as `save_path` within the Training Script if you'd like, but this isn't essential

### MultiPredNet Figure Script

Within `matlab_multiprednet_figures_showcase.m`:

The figure generation will display two test set results simultaneously

- [Line 8,9, 10, 11; ]: pass test data location to the `load()` functions
- [Lines 15, 16, 17; 29, 30, 31; 40, 41, 42 and 61]: pass output representations (from Inference Script's `save_path`) location to the `load()` functions

### Finally

- Save your changes to these files

## 3) Running the Experiment

Within Terminal or a suitable IDE:

1) Run `python_multiprednet_train_showcase.py`. Tensorflow will start up and run the training code. Console output should look as follows:

<img src="https://github.com/TomKnowles1994/MultiPredNet/blob/main/examples/readme_training_example.gif" width="600">

   Wait until console output stops

2) Run `python_multiprednet_gen_reps_showcase.py`. The inference process will begin. Console output should look as follows:

<img src="https://github.com/TomKnowles1994/MultiPredNet/blob/main/examples/readme_inference_example.gif" width="600">

   Wait until console output stops. Repeat this for each test set and sensor dropout you wish to output representations for.
   NOTE: you will need to run this a total of six times to generate the figure as shown below. This entails:
   
   - Running `python_multiprednet_gen_reps_showcase.py` on your 1st testset, with avail_modality (line 19) set to 'both'
   - Doing the same, but with avail_modality (line 19) set to 'visual'
   - Once more, but with avail_modality (line 19) set to 'tactile'
   
   - Then, repeat this for your 2nd testset

Within MATLAB:

3) Run `matlab_multiprednet_figures_showcase.m`. Figures will be generated showing the Representational Similarity Matrices. An example of the final output is shown below:

<img src="https://github.com/TomKnowles1994/MultiPredNet/blob/main/examples/readme_RSM_example.png" width="800">

A successful run on sample NRP data will have a score of 0.2-0.3, depending on the Test Set used. If manually collecting NRP data, results may be out of this range due to differences in experiment setup.
