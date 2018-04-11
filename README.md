# Information Retrieval Classification
Pytorch Implementation for sentiment analysis on product's comment

## Requirements
Please use Python 3, ubuntu 16.04, Git, and NVIDIA GPU with CUDA toolkit 8 and cUDNN 6. To install python library:
```
pip3 install -r requirements.txt
python -m spacy download en #for downloading English model for spaCy tokenizer
```
If failed please look requirements.txt and install one by one.

## Dataset
The dataset should be seperated into training file and test file in CSV format
When preparing CSV file, the dataset should not use indexing (if saving using pandas, use pd.to_csv(file_name, index=False)). There must be 2 columns: text and label.

## CNN and LSTM Model
The custom module of CNN and LSTM model are saved in model_module folder.

## JSON file for configuration training model
The JSON file are already prepared with appropriate setting. Most of the setting are for model hyperparameter.
Important parameter that might need to be change:
1. "train_dataset_path": train data path
2. "dev_dataset_path": test data path
3. "result_folder_path": where to save result such as model, image of confusion matrix, etc 
4. "use_git": whether to use current commit information for better result versioning. if true result_folder_path = result_folder_path/\[branch_name\]_\[commit_date_GMT_0\]_\[time_duration_after_commit\]
6. "pretrained_word_embedding_name": Name of pretrained word vectors used. For now, word2vec is recommended. word2vec can be downloaded at https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
7. "pretrained_word_embedding_path": Path of word vectors .bin file
8. "embedding_dim": Size of word vectors. 300 for word2vec
9. "train_embedding_layer": Set to false to make embedding layer static.
10. "epoch": number of training rounds.
11. "kernel_sizes" (CNN only): List of region sizes used fo CNN. Currently, ensemble learning is created by modifying this parameter.

Please look at the json file and try to run training first to understand more

## Running training to save model, plot evaluation accuracy, plot evaluation loss, confusion matrix, precision, recall, and F1 Score
CNN
```
python train_cnn.py --path train_cnn_parameter.json
```

LSTM
```
python train_lstm.py --path train_lstm_parameter.json
```

The result of confusion_matrix, precision, recall, and F1 score is displayed at console output.
The image of confusion_matrix are saved in result_folder_path/\[branch_name\]_\[commit_date_GMT_0\]_\[time_duration_after_commit\]/confusion_matrix_folder_path

## View evaluation accuracy and loss with tensorboard
Please run
```
tensorboard --logdir=[result_folder_path]
```
To view graph of test loss and accuracy

## Predicting Ensemble Learning and prediction time
Jupyter notebook, trained model, and vocabulary data are prepared for these tasks. Trained model and vocabulary data are saved in ensemble_learning_related. The Jupyter notebook is saved as Prediction_Time_and_Ensemble_Learning.ipynb. Please run Juypter server before running Juputer notebook.
```
jupyter notebook
```



