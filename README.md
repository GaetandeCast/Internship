This repository contains the experiments conducted during my internship at probabl in the open source team to work on the issues of MDI for random forest.

The `report` folder contains the internship report and oral defense material, as well as all the code used to produce results and illustrations.

Most of the code requires [this fork of scikit-learn](https://github.com/GaetandeCast/scikit-learn/tree/unbiased-feature-importance) that implements a Cython optimized version of the method UFI proposed by [Zhou et al. (2021)](https://dl.acm.org/doi/abs/10.1145/3429445).