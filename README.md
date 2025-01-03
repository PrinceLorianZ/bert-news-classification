### experimental environment
transformers
scikit-learn
python=3.8
pytorch==1.11.0
torchvision==0.12.0
torchaudio==0.11.0
cudatoolkit=11.3
### workflows
### Training documents
python tarin.py
### predict
python predict.py
### introduction
- 1. Data Processing Flow
The dataset consists of 10 categories, including sports, finance, real estate, home, education,
technology, fashion, current affairs, games, and entertainment. Each row of the dataset is organized as label + text.
The process of reading the dataset is as follows:
- a.First read the dataset data labels and content and stuff them inside the list.
- b.The text data and labels are converted into a data format suitable for input into the BERT model and 
returned as data items of the dataset. This involves the operation of binning the text and replacing the token_id list and the mask list with zeroes when the length of the text is less than the longest length of the text.
- c. Use the DataLoader object to load the dataset in batches of the specified batch_size, and support 
for disrupting the order of the data.

- 2. Model calling
Construct a text categorization model based on BERT. The model encodes the input through the BERT model and
maps the encoding results to category labels through the fully connected layer. During the training process, the parameters of the BERT model are fine-tuned.

- 3. Model parameters
Optimizer: AdamW is an optimizer algorithm based on gradient descent, which can adaptively adjust the learning rate, the
Loss Function: Cross Entropy Loss Function is a commonly used loss function for classification problems, which is used to measure the difference between the model predictions and the true labels.
As well as the batch, number of rounds, learning rate, and number of fully connected layer filters defined in the screenshot below

- 4. Classification results
Based on the weights of the pre-trained bert, 10 epochs were trained. the results are shown below:
The training process is shown in the following figure: acc accuracy is about 91.8% again, and the test set is around 86%.
