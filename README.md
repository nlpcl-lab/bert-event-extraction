# bert-event-extraction
Pytorch Solution of Event Extraction Task using BERT on ACE 2005 corpus

## Prerequisites

1. Prepare **ACE 2005 dataset**. 

    [nlpcl-lab/ace2005-preprocessing](https://github.com/nlpcl-lab/ace2005-preprocessing)
    
    ```
    ├── data
    │     └── test.json
    │     └── dev.json
    │     └── train.json
    │...
    ```

2. Install the packages.
   ```
   pip install pytorch==1.0 pytorch_pretrained_bert==0.6.1 numpy
   ```

## Usage

### Train
```
python train.py
```

### Evaluation
```
python eval.py
```

## Reference
* Jointly Multiple Events Extraction via Attention-based Graph Information Aggregation (EMNLP 2018), Liu et al. [[paper]](https://arxiv.org/abs/1809.09078)
* lx865712528's EMNLP2018-JMEE repository [[github]](https://github.com/lx865712528/EMNLP2018-JMEE)
* Kyubyong's bert_ner repository [[github]](https://github.com/Kyubyong/bert_ner)
