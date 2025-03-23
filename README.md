# SH-Project
Code for my Senior Honours Project, 'Distinguishing Transversely and Longitudinally Polarised Z Bosons Using Machine Learning'. This used a DNN to predict the polarisations of Z bosons, which was trained and tested on a dataset containing $ZH \rightarrow \ell\overline{\ell}\gamma \gamma$ events. This code was run in a Python 3.9.6 virtual environment, with requirements given in requirements.txt. The code is adapted from the repositories https://github.com/Satriawidy/HggNN and https://github.com/alexandramcadam/ZPolarisationNN/tree/main/DNN. 

# Implementation
This code was implemented as follows:
1. Process the raw dataset into a form that can be inputted into the DNN using process.py
2. Train the DNN using train.py
3. Test the DNN using test.py
4. Evaluate the DNN's performance on a variety of metrics using plots.py
5. Calculate and plot the likelihood curve using likelihood.py

