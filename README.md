# DBT-plasma-proteomic

This repository includes the code used in dual blocker therapy (DBT) plasma proteomic study.

**Plasma proteome profiling reveals dynamic of cholesterol marker after PD1/CTLA4 dual blocker therapy**

Jiacheng Lyu, Lin Bai, Yumiao Li, Xiaofang Wang, Zeya Xu, Tao Ji, Hua Yang, Zizheng Song, Zhiyu Wang, Yanhong Shang, Lili Ren, Yan Li, Aimin Zang, Youchao Jia, and Chen Ding

## Code overview
> The below figure numbers were corresponded to the paper version.

### 1. Figure1.ipynb
This Jupyter Notebook (with Python 3 kernel) contained the code for the proteomic analysis of DBT cohort design, quality control, data distribution, and biological processes description

Output figures and tables:  
* Figure 1C, S1A, S1B, S2A, S2B, S2C

### 2. Figure2.ipynb
This Jupyter Notebook (with Python 3 kernel) contained the code for the proteomic analysis of proteome and clinical indicators among healthy control, pre DBT and 1st DBT

Output figures:  
* Figure 2A-D, S3A, S3B

### 3. Figure3.ipynb
This Jupyter Notebook (with Python 3 kernel) contained the code for the  analysis of clinical indicators between disease non-progressive (DNP) and disease progressive (DP)

Output figures:  
* Figure 3, Figure S4

### 4. Figure4.ipynb
This Jupyter Notebook (with Python 3 kernel) contained the code for the  analysis of proteome between disease non-progressive (DNP) and disease progressive (DP)

Output figures:  
* Figure 4, Figure S5

### 5. Figure5.ipynb
This Jupyter Notebook (with Python 3 kernel) contained the code for the machine learning construction, model evaluation, and independent validation

Output figures:  
* Figure 5, Figure S8

## Environment requirement
The following package/library versions were used in this study:
* python (version 3.9.15)
* pandas (version 1.5.3)
* numpy (version 1.26.3)
* scipy (version 1.12.0)
* matplotlib (version 3.7.3)
* seaborn (version 0.11.2)
* scikit-learn (version 1.2.1)
* rpy2 (version 3.5.6)
* gprofiler (version 1.0.0)
* adjustText

## Folders Structure
The files are organised into four folders:
* *code*: contains the python code in the ipython notebook to reproduce all analyses and generate the the figures in this study.
* *document*: which contains all the proteomics and clinical patient informations required to perform the analyses described in the paper.
* *documents*: contains the related annotationfiles and the Supplementary Table produced by the code.
* *figure*: contains the related plots produced by the code.
