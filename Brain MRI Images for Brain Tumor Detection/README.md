# Brain MRI Images for Brain Tumor Detection

This project aims to develop and evaluate deep learning models for brain tumor classification using brain MRI images. It applies advanced preprocessing, model training, evaluation, and interpretability techniques to build robust and clinically relevant classifiers.

## Project Overview

- Comprehensive exploratory data analysis (EDA) on MRI image properties such as resolution, aspect ratio, brightness, contrast, sharpness, and noise.
- Data cleaning including identification and removal of corrupted or low-quality images.
- Implementation of multiple state-of-the-art convolutional neural network architectures:
  - ResNet-50
  - DenseNet-121
  - EfficientNet-B0
- Automated hyperparameter optimization using Optuna with cross-validation for each architecture.
- Training deep learning models with performance evaluated on test data using metrics including accuracy, precision, recall, AUC-ROC.
- Advanced explainability techniques such as Grad-CAM and Integrated Gradients to visualize important regions impacting model decisions.
- Error analysis to assess false positives and false negatives with visualization and tabular reports.
- Organized output of results including:
  - Confusion matrices and classification reports.
  - ROC and Precision-Recall curves.
  - Feature explainability plots.
  - Comprehensive comparison tables.
- Support for GPU acceleration if available.

## Getting Started

### Requirements

- Python 3.8+
- PyTorch & torchvision
- scikit-learn
- Optuna
- PIL/Pillow
- Matplotlib, Seaborn
- tqdm
- Other standard data science libraries (numpy, pandas)

### Installation

Install dependencies using pip:


### Dataset

- The project requires brain MRI image data labeled for tumor presence.
- Dataset preprocessing and splitting into train/test is handled within the pipeline.

### Usage

1. Prepare your dataset CSV with paths and labels.
2. Configure architectures to train/test (ResNet-50, DenseNet-121, EfficientNet-B0).
3. Run the main pipeline script to optimize hyperparameters, train models, and evaluate performance.
4. Results, plots, and explainability visualizations will be saved into the `results/` directory.

Basic training command example:


## Results

- Extracted metrics for all models include accuracy, precision, recall, and ROC AUC.
- Visualization of confusion matrices and ROC curves for direct model performance comparison.
- Grad-CAM and Integrated Gradients provide heatmaps showing model focus areas on MRI scans, aiding interpretability.
- Detailed error analysis reporting false positives and false negatives.

## Project Structure

- `brain_mri_images_for_brain_tumor_detection_draft2.py`: Main script with pipeline implementation.
- `results/`: Output directory containing metrics, plots, tables, and heatmaps.
- Organized folders inside `results/` for model-specific outputs and advanced explainability.

## Future Work

- Incorporate Docker containerization and model serving APIs for production deployment.
- Extend to multi-class tumor subtype classification.
- Integrate with cloud platforms for scalable training and real-time inference.
- Further research into improved interpretability and uncertainty quantification.

## Acknowledgements

This project leverages open-source deep learning libraries and interpretability methods to advance brain tumor detection research.

---

Feel free to contribute or raise issues to improve the project!

