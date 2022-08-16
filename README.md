Hello, this is the official repository of my Master's Thesis @Jacobs University
# DeepSNPs: Deep Learning for Single Nucleotide Polymorphism Disease Classification based on Chromosome Location"

# Raw Datasets
  - GWAS Catalog
  - DisGeNET
  - [CTCF Binding Sites](https://insulatordb.uthsc.edu)

# Ready-to-go
- [Filtered (preprocessed) datasets used for to build (train) DeepSNPs](https://drive.google.com/drive/folders/1m_1X-9Fy9SXP9jvZN0hUdXw4DmaARj_h?usp=sharing)
- [Official Training Notebook](https://drive.google.com/file/d/1JTClPuI-N3tDKpNdfw36vFYHR5yNQqgX/view?usp=sharing)

# Deep Learning Model: Bidirectional LSTM using Keras Tensorflow
- Overall Workflow Picture

![](https://github.com/bonaventuredossou/deepsnpsthesis/blob/main/pictures/DiagramDeepSNPs.jpg)


- Models's weights could be accessed [here](https://drive.google.com/file/d/1-ENOpruomCh9kE9nPuBtw-gxgwp-ayFh/view?usp=sharing)

- Results:
    - Training Loss (Accuracy): 0.1947 (0.9016)
    - Validation Loss (Accuracy): 1.5455 (0.7151)
    - Testing Loss (Accuracy): 0.1963 (0.9036)
    - AUC ROC: 0.8733
  


- Classification Report

![](https://github.com/bonaventuredossou/deepsnpsthesis/blob/main/pictures/classreport.png)


- Confusion Matrix

![](https://github.com/bonaventuredossou/deepsnpsthesis/blob/main/pictures/conf_matrix_best.png)



- AUC ROC Curve

![](https://github.com/bonaventuredossou/deepsnpsthesis/blob/main/pictures/auc.png)
  

- Features Importance with correct predictions

![](https://github.com/bonaventuredossou/deepsnpsthesis/blob/main/pictures/features_importance_correct_predictions.png)
  


- Features Importance with wrong predictions

![](https://github.com/bonaventuredossou/deepsnpsthesis/blob/main/pictures/features_importance_wrong_predictions.png)
