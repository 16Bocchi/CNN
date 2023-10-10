# 32513 31005 Advanced Data Analytics Algorithms, Machine Learning Assignment 2
### By Daniel Braithwaite

## Details:
For this assignment, I have decided to do option 2, which is to **"Build a machine learning system as a solution to a practical challenge"**.

I have therefore decided to develop and train a system using the Modified National Institute of Standards and Technology (MNIST) database [available here:](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) <br>
The MNIST database contains 60,000 training and 10,000 test images of normalised 28x28 pixel anti-aliased numbers (0-9), for which I will be using to train a Convoluted Neural Network (CNN) to recognise these digits.
The format of these files is a CSV document (too big to include in repository), with each row consisting of 785 values. The first value is the label (0-9) and the remaining 784 values are the greyscale pixel values (0-255).

## Math:
Consider a matrix _X_ of _M_ rows, with each row having 784 columns (for each pixel). <br>
Transpoing this matrix _X_ will return a matrix with _M_ columns and 784 rows.
