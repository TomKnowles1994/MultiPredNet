# MultiPredNet

A multimodal predictive coding implementation for place recognition. Suitable datasets can be gathered from the Neurorobotics Platform and used to train the predictive coding network to generate suitable representations of the robot's environment. This can then be used in other tasks, such as navigation. Quality of the learned representations is determined its Representational Similarity Matrix; a good learned representation is taken to be one that changes within itself in a similar way to other unambiguous representations, such as pose space (x, y, theta). This repository also contains scripts for generating and comparing these matrices.

## Dependencies:

- Python 3.5 or higher
- Tensorflow 2.0 or higher
- MATLAB 2020b with valid License
- MATLAB Add-ons installed: Robotic Systems Toolbox and Signal Processing Toolbox
- ROS (any version)

## Also required:

- A ROSbag file, generated from the NeuroRobotics Platform Whiskeye Experiment.

**OR**

- Alternatively, sample ROSbag and MATLAB files are available at **dataset_link_awaiting_curation**

NB: If your data has already been unpacked from its ROSbag, then ROS nor the ROSbag is required. Use the generated .mat files instead

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

### 3) Running the Experiment

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

A successful run on NRP data will have a score of 0.2-0.3, but anything above 0.1 is considered significant.
