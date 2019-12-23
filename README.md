# bert-event-extraction
Pytorch Solution of Event Extraction Task using BERT on ACE 2005 corpus

## Prerequisites

1. Prepare **ACE 2005 dataset**. 

2. Use [nlpcl-lab/ace2005-preprocessing](https://github.com/nlpcl-lab/ace2005-preprocessing) to preprocess ACE 2005 dataset in the same format as the [data/sample.json](https://github.com/nlpcl-lab/bert-event-extraction/blob/master/data/sample.json). Then place it in the data directory as follows:
    ```
    ├── data
    │     └── test.json
    │     └── dev.json
    │     └── train.json
    │...
    ```

3. Install the packages.
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
python eval.py --model_path=latest_model.pt
```

## Result	

### Performance	

<table>	
  <tr>	
    <th rowspan="2">Method</th>	
    <th colspan="3">Trigger Classification (%)</th>	
    <th colspan="3">Argument Classification (%)</th>	
  </tr>	
  <tr>	
    <td>Precision</td>	
    <td>Recall</td>	
    <td>F1</td>	
    <td>Precision</td>	
    <td>Recall</td>	
    <td>F1</td>	
  </tr>	
  <tr>	
    <td>JRNN</td>	
    <td>66.0</td>	
    <td>73.0</td>	
    <td>69.3</td>	
    <td>54.2</td>	
    <td>56.7</td>	
    <td>55.5</td>	
  </tr>	
  <tr>	
    <td>JMEE</td>	
    <td>76.3</td>	
    <td>71.3</td>	
    <td>73.7</td>	
    <td>66.8</td>	
    <td>54.9</td>	
    <td>60.3</td>	
  </tr>	
  <tr>	
    <td>This model (BERT base)</td>	
    <td>63.4</td>	
    <td>71.1</td>	
    <td>67.7</td>	
    <td>48.5</td>	
    <td>34.1</td>	
    <td>40.0</td>	
  </tr>	
</table>	

The performance of this model is low in argument classification even though pretrained BERT model was used. The model is currently being updated to improve the performance.

## Reference
* Jointly Multiple Events Extraction via Attention-based Graph Information Aggregation (EMNLP 2018), Liu et al. [[paper]](https://arxiv.org/abs/1809.09078)
* lx865712528's EMNLP2018-JMEE repository [[github]](https://github.com/lx865712528/EMNLP2018-JMEE)
* Kyubyong's bert_ner repository [[github]](https://github.com/Kyubyong/bert_ner)
