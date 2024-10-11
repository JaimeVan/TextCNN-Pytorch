import argparse

import torch
import torch.optim as optim

from model import textCNN as myModel
import expirement

from dataset_loader import get_text_dataloader

#%%

def main():
    print("Pytorch Version:", torch.__version__)
    parser = argparse.ArgumentParser(description='TextCNN')
    #Training args
    parser.add_argument('--data_csv', type=str, default='/home/fanjm/PythonPrj/Pytorch-TextCNN/data/smp2017.csv',
                        help='file path of training data in CSV format (default: ./train.csv)')
    
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 10)')
    
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    
    parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 64)')
    
    parser.add_argument('--kernel_height', type=str, default='3,4,5',
                    help='how many kernel width for convolution (default: 3, 4, 5)')
    
    parser.add_argument('--out_channel', type=int, default=100,
                    help='output channel for convolutionaly layer (default: 100)')
    
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate for linear layer (default: 0.5)')
    
    parser.add_argument('--num_class', type=int, default=31,
                        help='number of category to classify (default: 2)')
    
    #if you are using jupyternotebook with argparser
    # args = parser.parse_known_args()[0]
    args = parser.parse_args()
    
    
    #Use GPU if it is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    #%% Split whole dataset into train and valid set
    label_dict = {
        'app': 0, 'bus': 1, 'calc': 2, 'chat': 3, 'cinemas': 4, 'contacts': 5, 
        'cookbook': 6, 'datetime': 7, 'email': 8, 'epg': 9, 'flight': 10, 
        'health': 11, 'lottery': 12, 'map': 13, 'match': 14, 'message': 15, 
        'music': 16, 'news': 17, 'novel': 18, 'poetry': 19, 'radio': 20, 
        'riddle': 21, 'schedule': 22, 'stock': 23, 'telephone': 24, 'train': 25, 
        'translation': 26, 'tvchannel': 27, 'video': 28, 'weather': 29, 'website': 30
    }
    

    #%%Show some example to show the dataset
    train_loader, valid_loader, myVocabModel = get_text_dataloader(args.data_csv, label_dict, test_ratio=0.3, batch_size=args.batch_size, vector_size=1000)
    vocab = myVocabModel.get_vocab_from_w2v()
    #%%Create
    
    kernels = [int(x) for x in args.kernel_height.split(',')]
    m = myModel(vocab, args.out_channel, kernels, args.dropout , args.num_class).to(device)
    # print the model summery
    print(m)
        
    best_test_acc = -1
    
    #optimizer
    optimizer = optim.Adam(m.parameters(), lr=args.lr)
    
    for epoch in range(1, args.epochs+1):
        #train loss
        tr_loss, tr_acc = expirement.train(m, device, train_loader, optimizer, epoch, args.epochs)
        print('Train Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, tr_loss, tr_acc))
        
        ts_loss, ts_acc = expirement.valid(m, device, valid_loader)
        print('Valid Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, ts_loss, ts_acc))
        
        if ts_acc > best_test_acc:
            best_test_acc = ts_acc
            #save paras(snapshot)
            print("model saves at {}% accuracy".format(best_test_acc))
            torch.save(m.state_dict(), "best_validation")
            

if __name__ == '__main__':
    main()

