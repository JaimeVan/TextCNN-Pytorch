# TextCNN Pytorch

## Brief
TextCNN implement in new version Pytorch and torchtext. We used the latest version of torchtext to implement a simple task of Chinese classification. It is worth noting that torchText development is stopped and the 0.18 release (April 2024) will be the last stable release of the library.

## Getting Started

### 1.requirements
```
python      3.10.15
jieba       0.42.1
torch       2.3.0
torchtext   0.18.0
numpy       1.26.4
```

### 2.Prepare dataset and pretrained vocaulary
You should make your onw dataset directory in the workshop directory such ./data/dataset.csv prepare data as such format:

|text|label|
|--|--|
|怎么分担|chat|
|美国最近有什么新闻|news|

And prepare pretrained vocabulay in directory such as ./embedding/sgns.zhihu.word, you can download chinese vocabulary in [this github repo](https://github.com/Embedding/Chinese-Word-Vectors)

### 3.Prepare your own label dict

You should prepare you own label_dict in main.py to fit your dataset such as:
```python
label_dict = {
    'app': 0, 'bus': 1, 'calc': 2, 'chat': 3, 'cinemas': 4, 'contacts': 5, 
    'cookbook': 6, 'datetime': 7, 'email': 8, 'epg': 9, 'flight': 10, 
    'health': 11, 'lottery': 12, 'map': 13, 'match': 14, 'message': 15, 
    'music': 16, 'news': 17, 'novel': 18, 'poetry': 19, 'radio': 20, 
    'riddle': 21, 'schedule': 22, 'stock': 23, 'telephone': 24, 'train': 25, 
    'translation': 26, 'tvchannel': 27, 'video': 28, 'weather': 29, 'website': 30
}
```

### 4.Run the code
```bash
python main.py --data_csv ./data/dataset.csv --epochs 10 --lr 0.01 --batch_size 64 --kernel_height 3,4,5 --out_channel 100 --dropout 0.5 -testdata_ratio 0.3
```

## Reference

Thanks to the information from following sources:
- https://github.com/leohsuofnthu/Pytorch-TextCNN
- https://github.com/delldu/TextCNN
- https://github.com/Embedding/Chinese-Word-Vectors
