import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN_Tfidf_remove_embedding(nn.Module):
    def __init__(self, dim_channel, kernel_wins, dropout_rate, num_class):
        super(TextCNN_Tfidf_remove_embedding, self).__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, dim_channel, (w, 1)) for w in kernel_wins
        ])
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc = nn.Linear(len(kernel_wins) * dim_channel, num_class)
        
    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(3)
        
        con_x = [F.relu(conv(x)) for conv in self.convs]
        
        pool_x = [F.max_pool2d(cx, (cx.size(2), 1)).squeeze(3).squeeze(2) for cx in con_x]
        
        fc_x = torch.cat(pool_x, dim=1)
        
        fc_x = self.dropout(fc_x)
        
        logit = self.fc(fc_x)
        
        return logit

#%% Text CNN model
class textCNN(nn.Module):
    
    def __init__(self, vocab_built, dim_channel, kernel_wins, dropout_rate, num_class):
        super(textCNN, self).__init__()
        
        emb_dim = len(next(iter(vocab_built.values()))) 
        
        self.embed = nn.Embedding(len(vocab_built), emb_dim, dtype=torch.float32)
        
        # comment to use random data in embedding layer
        # embeddings = torch.tensor(list(vocab_built.values()), dtype=torch.float32)
        # self.embed.weight.data.copy_(embeddings)
        
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc = nn.Linear(len(kernel_wins) * dim_channel, num_class)
    
    def forward(self, x):
        emb_x = self.embed(x)
        emb_x = emb_x.unsqueeze(1)

        con_x = [F.relu(conv(emb_x)) for conv in self.convs]  # [batch_size, dim_channel, seq_length]

        pool_x = [F.max_pool2d(cx, (cx.size(2), 1)).squeeze(3).squeeze(2) for cx in con_x]
        fc_x = torch.cat(pool_x, dim=1)  # [batch_size, len(kernel_wins) * dim_channel]
        
        fc_x = self.dropout(fc_x)
        
        logit = self.fc(fc_x)  # [batch_size, num_class]
        
        return logit