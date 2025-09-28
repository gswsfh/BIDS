# BIDS
Code and models from the paper "BIDS: A Reliable Boundary-based Intrusion Detection System".

We do not provide datasets as they are sourced from publicly available resources. They originate from: 
- https://www.unb.ca/cic/datasets/ids-2017.html
- https://www.unb.ca/cic/datasets/ids-2018.html
- https://www.unb.ca/cic/datasets/cic-unsw-nb15.html 

Please replace the dataset paths with those relevant to your project before running.

## Dependencies
pip install -r requirements.txt

## Related models
We provide relevant machine learning detection models in the MLModel folder, including: decision trees, SVM, Bayesian networks, KNN, logistic regression, and more.

We also provide relevant deep learning detection models in the DLModel folder, including: Convolutional Neural Networks, Attention Neural Networks, Standard Autoencoders, Variational Autoencoders, and Feature-Based Variational Autoencoders.

## BAE
The model is located in BAE.py, where we provide code for training, prediction, and visualization.

## Note
We have not provided the code for converting models to rules, as it is relatively straightforward.
