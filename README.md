**Python Coding Samples - Mia Patil**

**Overview**
This submission includes two Python projects that demonstrate my skills in data analysis, algorithm development, and quantitative modeling. The projects were completed as part of my coursework and showcase my ability to work with data structures, apply mathematical techniques, and visualize results.

**Project 1: Principal Component Analysis (PCA) for Image Reconstruction**
File: PCA.py

Description: This project implements Principal Component Analysis (PCA) to project and reconstruct images in a lower-dimensional space. It involves calculating eigenvalues and eigenvectors from a covariance matrix, projecting an image onto the principal components, and reconstructing the image. Additional functionalities include perturbing the projected image using Gaussian noise and combining the projection coefficients of two images to generate a convex combination.

Skills Demonstrated:
  Matrix operations and eigenvalue decomposition using NumPy.
  Data manipulation and mathematical concepts such as PCA and covariance matrices.
  Visualizing image reconstruction using Matplotlib.
  Efficient Python code with a focus on numerical computation.
  
How to Run:
  Make sure the dataset file (e.g., face_dataset.npy) is in the same directory as PCA.py.
  This dataset is quite large. I have provided a google drive link to download it: https://drive.google.com/file/d/1Og-sdEnE3bOrme3NXQxBXilxJqgsunzz/view?usp=sharing
  Run the project using the provided Makefile (run “make”)
  The script will run the tests for PCA projection, perturbation, and combination, displaying results with Matplotlib.

  
**Project 2: Hierarchical Agglomerative Clustering (HAC)**
File: HAC.py

Description: This project implements Hierarchical Agglomerative Clustering (HAC) from scratch. It uses a bottom-up clustering approach, where each data point starts as its own cluster, and clusters are merged iteratively based on minimum pairwise distances. The result is visualized using a dendrogram.

Skills Demonstrated:
  Data analysis by calculating feature vectors from socioeconomic data.
  Distance matrix computation using NumPy.
  Implementation of clustering algorithms without external libraries (other than NumPy and Matplotlib).
  Data visualization through a dendrogram plot to show the hierarchical structure of the clusters.
  
How to Run:
  Ensure the dataset file (e.g., socioeconomic_data.csv) is in the same directory as HAC.py.
  Run the project using the provided Makefile (run “make”)
  The script will prompt you to enter the number of data points (n) you'd like to cluster. Enter a value, and the HAC algorithm will run on    the first n rows of the data, displaying a dendrogram.
  
**Dependencies: both projects require the following Python libraries:**
NumPy, Matplotlib, SciPy (for PCA.py)
You can install the required dependencies using pip: pip install numpy matplotlib scipy

**Contact**
If you have any questions or need further clarifications, feel free to reach out to me at miapatil123@gmail.com.



