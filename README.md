# Classifying Stream and Ditch Channels Mapped from High-Resolution Digital Elevation Models Using Machine Learning
Busarello, M., Ågren, A., & Westphal, F., Lidberg, W.
<img width="4551" height="2943" alt="workflow" src="https://github.com/user-attachments/assets/df16ce8b-7919-4fd8-ac5c-330d8527b178" />

We have trained machine learning models to classify ditches and stream channels using hydrological features. The architecture used was XGBoost, and the channels were previously mapped using a U-Net model, slope, and sky-view factor.

This repository contains the U-Net model used to map the water channels, XGBoost models to classify water channels, the DEMs to calculate the hydrological and topographical features, and the ground truth polylines. The code available was used to train the models, predict the classes in new data, and perform all the pre-processing and processing steps to reproduce our study. If needed, future updates can be found here.

The channel network was split into different segment sizes to train the models: 10 m, 25 m, 50 m, 100 m, and unsplit. All of them can be created using the code present in this repository. The model trained with the 50 m segment size (hybrid) was chosen as the most balanced, and its performance can be evaluated in the image below. The U-Net mapped network was improved by a more accurate and robust classification of ditches and streams from the XGBoost model, also removing most of the false positives successfully.

<img width="1914" height="660" alt="comparison" src="https://github.com/user-attachments/assets/d76a6466-3fa1-4d4a-9acf-e71bc403d950" />

The data for this study comes from 12 study areas spread across Sweden, with different characteristics regarding land use and forest cover, among others. The DEMs were calculated from the laser data, which comes from Lantmäteriet. The data is originally organized into tiles of 2500 m x 2500 m, later being split into chips of 250 m x 250 m to train the U-Net model. The tiles were mosaiced to calculate the hydrological features needed to train the XGBoost model.

<img width="808" height="916" alt="map" src="https://github.com/user-attachments/assets/42fc6401-6029-4ec7-aae1-6bf2bdb19210" />
