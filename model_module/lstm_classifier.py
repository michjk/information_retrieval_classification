import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    '''
    LSTMClassifier is classification module based on LSTM
    Args:
        embedding_dim (int): The size of word vector.
        vocab_size (int): The number of words/tokens in vocabulary.
        label_size (int): The number of possible labels.
        hidden_dim: The number of features in the hidden state h of QRNN
        pretrained_embedding_weight (torch.Tensor): The pretrained word vectors (optional).
        train_embedding_layer (bool): Whether to train embedding layer or let embedding layer to be fixed.
        num_layers (int): The number of layers of QRNN.
        dropout (float): The probability of dropout in QRNN and dropout layer.
        use_gpu (bool): Whether to use GPU or not
    
    Inputs: x:
        - x (seq_len, batch, input_size): tensor containing the features of the input sequence.
    
    Output: logsoftmax
        - logsoftmax (batch, label_size) : tensor result of log softmax
    '''
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, pretrained_embedding_weight = None, train_embedding_layer = True,num_layers = 1, dropout = 0, use_gpu = True):
        super().__init__()

        #initialize properties
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        
        ## create nn module
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if not (pretrained_embedding_weight is None): #use pretrained word vectors
            self.word_embeddings.weight.data = pretrained_embedding_weight
            self.word_embeddings.weight.requires_grad = train_embedding_layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.hidden_to_label = nn.Linear(hidden_dim, label_size)

        #use gpu
        if use_gpu:
            self.cuda()
        
    def init_hidden(self):
        """
        Initialize weight of hidden
        """
        # the first is the hidden h
        # the second is the cell  c
        h = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)
        c = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)

        if self.use_gpu:
            h = h.cuda()
            c = c.cuda()
        
        self.hidden = (autograd.Variable(h),
                        autograd.Variable(c))
        
    def forward(self, sentence):
        self.batch_size = sentence.data.shape[1]
        self.init_hidden()
        
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        out, self.hidden = self.lstm(x, self.hidden)
        out = self.dropout(out)
        y = self.hidden_to_label(out[-1])
        log_probs = F.log_softmax(y)
        return log_probs


