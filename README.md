# Dopamine and DBS accelerate the neural dynamics of volitional action in Parkinson’s disease

This repository contains the code used for analyses in the paper _Dopamine and
DBS accelerate the neural dynamics of volitional action in Parkinson’s disease_,
which has been released as a preprint at bioRxiv:

https://www.biorxiv.org/content/10.1101/2023.10.30.564700v4

To get started right away, skip down to the section
[Getting started](#getting-started).

### Python

Most analyses were performed using Python and can in principle be done on any
standard computer running Windows, macOS or Linux. Most code was run on a
Windows 11 notebook with a 4-core CPU (Intel Core i7), 32GB RAM, without
dedicated GPU. The main Python code for analyses was packaged for ease of use
and can be found in the [_src/motor_intention_](src/motor_intention/) folder.
The analysis code makes use of other custom packages that were created and
released on PyPi for this purpose, including
[PTE](https://github.com/richardkoehler/pte),
[PTE Decode](https://github.com/richardkoehler/pte-decode),
[PTE Stats](https://github.com/richardkoehler/pte-stats). A stripped-down
version of
[py_neuromodulation](https://github.com/neuromodulation/py_neuromodulation) was
used for calculation of features for the machine learning-based decoding of
motor intention and can be found here:
https://github.com/richardkoehler/py_neuromodulation/tree/paper_motor_intenion.

### Granger causality

Computationally expensive analyses related to calculation of
[multivariate Granger causality in MNE-Connectivity](https://mne.tools/mne-connectivity/stable/auto_examples/granger_causality.html)
were performed on the
[HPC for Research cluster of the Berlin Institute of Health](https://hpc-docs.cubi.bihealth.org/).
Corresponding Python code and
[Slurm](https://slurm.schedmd.com/documentation.html) scripts can be found in
the folder [_granger_causality_](granger_causality/).

### MATLAB

Visualization and statistics related to Granger causality analyses were done in
MATLAB. The corresponding code can be found in the folder [_matlab_](matlab/).

## Getting started

To make sure the analysis steps are reproduced as close as possible, I recommend
following the steps below:

First, download this repository using the version control system
[git](https://git-scm.com/). Type the following command into a terminal:

```bash
git clone https://github.com/richardkoehler/paper-motor-intention
```

Use the package manager
[conda](https://docs.conda.io/projects/conda/en/latest/index.html) to set up a
new working environment. To do so, in your command line navigate to the location
where this repository is stored on your machine and type:

```bash
conda env create -f env.yml
```

This will create a new conda environment called `motor-intention` and install
Python 3.10.9 including all necessary packages. Then activate the environment:

```bash
$ conda activate motor-intention
```

<!-- [![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link] -->

<!-- [![GitHub Discussion][github-discussions-badge]][github-discussions-link] -->

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
<!-- [actions-badge]:            https://github.com/richardkoehler/paper-motor-intention/workflows/CI/badge.svg
[actions-link]:             https://github.com/richardkoehler/paper-motor-intention/actions -->
<!-- [github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/richardkoehler/paper-motor-intention/discussions -->
<!-- [rtd-badge]:                https://readthedocs.org/projects/paper-motor-intention/badge/?version=latest
[rtd-link]:                 https://paper-motor-intention.readthedocs.io/en/latest/?badge=latest -->

<!-- prettier-ignore-end -->
