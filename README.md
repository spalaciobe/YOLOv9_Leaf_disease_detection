# YOLOv9 Custom Training and PlantVillage Transformation

*This project is part of ongoing research on plant disease detection. A detailed academic paper covering the methodology, experiments, and results will be published soon. Stay tuned for the official release!*

This repository provides a comprehensive workflow for applying computer vision techniques to plant disease detection using YOLOv9 models. The project focuses on training these models, transforming the PlantVillage dataset for object detection, and evaluating their performance.

By leveraging deep learning and image processing methodologies, this project aims to improve the accuracy and efficiency of detecting plant diseases, which is essential for sustainable agriculture and food security. The following Jupyter notebooks detail the processes involved in customizing model training, transforming existing datasets, and rigorously testing and evaluating model performance.

## Model Inferences

  <img src="images/Figure 3.jpg" alt="Inference 3" width="600">
  <img src="images/Figure 4.jpg" alt="Inference 4" width="600">

## Notebooks Overview

1. **Custom YOLOv9 Training** - [`training-yolov9-t.ipynb`](training-yolov9-t.ipynb): This notebook details the training process of YOLOv9 on a custom tomato leaf disease dataset. It includes data augmentation techniques, model architecture configuration, hyperparameter tuning, and training logs to track the training progress.

2. **PlantVillage Transformation** - [`training-leaf.ipynb`](training-leaf.ipynb): This notebook focuses on converting the PlantVillage classification dataset into an object detection format. It covers the semi-automated process of bounding box labeling, ensuring the dataset is structured correctly for YOLOv9 training. Additionally, it discusses the handling of annotations and the dataset split into training and validation sets.

3. **Model Testing and Evaluation** - [`models-test.ipynb`](models-test.ipynb): This notebook evaluates the performance of the trained YOLOv9 models on both the custom and transformed datasets. It compares metrics such as precision, recall, and mean Average Precision (mAP) to assess the models' accuracy and efficiency in detecting plant diseases.

## Custom Training Workflow

- **Data Augmentation**: The workflow includes the expansion of the "Healthy" class and various transformations (e.g., grayscale, zoom) to enhance the robustness of the dataset.
- **Model Configuration**: Different YOLOv9 models were trained with optimized parameters tailored for effective disease detection.

## PlantVillage Dataset Transformation

- **Annotation Process**: The notebook implements an automatic leaf detection method and assigns class labels based on the folder structure. Manual corrections are performed to ensure the accuracy of the annotations.
  <img src="images/Figure 2.jpg" alt="Annotation process" width="600">

- **Dataset Structuring**: The transformation process prepares the dataset for YOLOv9 by converting annotations and splitting the data into training and validation sets.

## Model Testing and Evaluation

- **Metrics**: The evaluation includes precision, recall, and mAP, focusing on both detection accuracy and inference speed.
- **Cross-Dataset Testing**: The models are tested on both complex (custom) and simpler (PlantVillage) images to demonstrate their generalization capabilities across different datasets.

## Datasets

This project utilizes the following datasets for training and evaluation:

1. **Custom Tomato Leaf Disease Dataset**: Used to train the models. [Kaggle Link](https://www.kaggle.com/datasets/sebastianpalaciob/tomato-leaf-diseases-dataset-for-object-detection)
2. **PlantVillage Dataset**: Obtained from PlantVillage Dataset Transformation [Kaggle Link](https://www.kaggle.com/datasets/sebastianpalaciob/plantvillage-for-object-detection-yolo)


---

