![image](.github/banner.png)

# Lung Tumor Detection for the Duke Quantitative Image and Analsysis Lab (QIAL) U24 Precinical Trail

This repository contains methods for training a V-net to detect lung nodules in micro-CT images of mice. The purpose of this project is to create tools which may be used in computer aided diagnosis (CAD) of scans performed in the preclinical arm of a co-clinical trial.

This repository is still under contrustion.

## Project breakdown

This work contains three princple components:

1. Data generation
2. Network training
3. Analysis

### Data generation

This work relies on using simulated and real datasets of tumor bearing mice. A major component of this project was to create simulated datasets with similar properties and features as those found in the real sets. The code for generating these simulated scans is found under the `Simulate_data` folder.

### Network training and analysis

The network and analysis code are found under the `Networks` and `Analysis` folders respectively. Additional information on the contents of these folders can be found under their respective folders.