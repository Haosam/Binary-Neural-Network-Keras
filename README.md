# Binary-Neural-Network-Keras
## A Keras code on Binary Neural Networks
Files to run in this order<br/>
Step 1: Run BNN_training.py<br/>
This will give you your training for a Fully Connected 784-512-512-10 MLP layer.<br/>
<br/>
Step 2: Go into models folder. Run weight_extract.py<br/>
This will give you your binarized weights on your final trained mode constrained to -1 and +1 <br/>
Please edit the code for the binarization of your layers. The original code is run on 4 hidden layers and will not work with the current BNN_training.py<br/>
<br/>
Step 3: Either run BNN_new_train.py for evaluation of your saved weights loaded into a new model<br/>
OR<br/>
BNN_train_fun.py to test the trained results on real-time to see whether the model is able to predict your results<br/>
<br/>
Miscellanous files (Do not Remove):<br/>
- activations.py
- binary_layers.py
- bnn_ip_binary.py (This is for fun to visualize the actual binarization of the input images)
