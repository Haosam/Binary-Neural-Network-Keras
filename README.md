# Binary-Neural-Network-Keras
## A Keras code on Binary Neural Networks
Files to run in this order<br/>
Step 1: Run BNN_full_binary_io.py<br/>
This will give you your training for a Fully Connected 784-512-512-10 MLP layer. Binarization fo final weights is done at the end of training. Please edit the code for the binarization of your layers, if you added more layers<br/>
Step 1.1: Run Mnist_cnn.py<br/>
This will run the CNN model. BatchNorm and dropout has been commented out <br/>
<br/>
## Updated:
Step 2: Go into models folder. Run weight_extract_cnn.py (Run this if you ran mnist_cnn.py)<br/> 
This will give you your binarized weights and conv layers on your final trained mode constrained to -1 and +1 <br/>
Please edit the code for the binarization of your layers.<br/>
<br/>
Step 3: (Results_validate folder) Either run BNN_new_train.py for evaluation of your saved weights loaded into a new model<br/>
OR<br/>
BNN_train_fun.py to test the trained results on real-time to see whether the model is able to predict your results<br/>
<br/>
## Note: Please Edit your PATH files
Miscellanous files (Do not Remove):<br/>
- activations.py
- binary_layers.py <br/>
(Results_validate Folder) <br/>
- bnn_ip_binary.py (This is for fun to visualize the actual binarization of the input images)
