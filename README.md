# Analysis and Data Mining of a Medical Database.



My university project focuses on extracting valuable insights from a medical database using three powerful techniques: Principal Component Analysis (PCA), K-means clustering, and Hierarchical Ascendant Classification (HAC).

PCA helps us identify patterns and reduce the dimensionality of the data, making it easier to understand and visualize complex relationships among medical variables.
K-means clustering further enhances our analysis by grouping similar medical records together, allowing us to uncover meaningful clusters or subgroups within the patient population.
HAC complements the clustering approach by providing a hierarchical perspective, offering a more nuanced understanding of how patient records can be grouped into clusters based on their similarities. This method organizes data into a tree-like structure, making it easier to visualize different levels of similarity among patients.
## TRY IT BY YOURSEL

https://projectafd.streamlit.app/
(you can use the file named cancer_de_poumon as an exemple)


 Project Objectives and Implementation Protocol

The objective of this project is to apply data analysis and mining methods to help doctors in their decision-making process. A medical database (`cancer_des_poumons.csv`) is provided to detect anomalies.

# Project Description

## 3.1 Data Preparation

![image](https://github.com/user-attachments/assets/1fcc163b-3f50-4642-8b41-ba6aa4666ab5)


This phase involves pre-processing and transforming raw data for further analysis. Steps include handling missing values, encoding, normalization, and correlation analysis.

###handling missing values and encoding
![image](https://github.com/user-attachments/assets/abd581af-378c-4d04-a50d-93e9475a94eb)

### normalization

![image](https://github.com/user-attachments/assets/d446df05-c9ec-4063-920a-73642e730fe8)

### correlation analysis

![image](https://github.com/user-attachments/assets/24a2d15f-1e00-4708-b0c1-7d9544a23419)





## 3.2 Feature Extraction
Principal Component Analysis (PCA) will be used to extract new factors and reduce dimensionality for better data representation.


![image](https://github.com/user-attachments/assets/e577e40f-7e61-4015-8f4a-b7918303fd35)


![image](https://github.com/user-attachments/assets/f54e93e6-a1c0-48b2-bb3b-b9e7bf386549)


### correlation circle
![image](https://github.com/user-attachments/assets/93e4af13-a285-478f-8417-dc164198aa0b)




## 3.3 Data Mining
Clustering methods, such as K-means and Hierarchical Ascendant Classification (HAC), will be applied to distinguish between healthy and sick patients.


### K-means

![image](https://github.com/user-attachments/assets/767223f3-873c-4191-8efb-5f3bf5542f68)

### Hierarchical Ascendant Classification (HAC)

![image](https://github.com/user-attachments/assets/388f51c7-a48f-49f1-aaa4-ce2803db4078)



# Setup
pip install -r requirements.txt

streamlit run app.py 



