# Adversarial Semantic Collisions
This repo contains implementation for EMNLP 2020 paper: 
[Adversarial Semantic Collisions](http://www.cs.cornell.edu/~shmat/shmat_emnlp20.pdf).

### Dependencies
The code is tested on Python 3 with torch==1.4.0 and transformers==2.8.0. 
Other requirements can be found in `requirements.txt`.

### Datasets and Models
We considered four tasks in this paper. The data and models can be downloaded from [here](https://zenodo.org/record/4263446#.X6iYUnVKjCJ) (the decompressed file can take upto 18GB of disk space).
Please extract the data and models into `COLLISION_DIR` defined in `constant.py`.

* For paraphrase identification task, the models are trained with HuggingFace example [scripts](https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py).
* For response suggestions task, the models are collected from [ParlAI](https://parl.ai/projects/polyencoder/).
* For document retrieval task, the models are collected from [Birch](https://github.com/castorini/birch).
* For extractive summarization task, the models are collected from [PreSumm](https://github.com/nlpyang/PreSumm).


### Language Models for Natural Collisions
For generating natural collisions (see Section 4.2.2 in our paper), we need to train language models (LMs) with the same
vocabulary as the target models we are attacking.  
We provide pre-trained LMs in the download link above and their training scripts in `scipts/` folder. 
LMs are fine-tuned from BERT or [Poly-encoder](https://arxiv.org/pdf/1905.01969.pdf) on [WikiText-103](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/).


### Generating Semantic Collisions
Now we can run collision attacks on the test set for the four tasks.
We provide example scripts for as following, where (A), (R), (N) denotes aggressive, 
regularized and natural collisions respectively.

**Paraphrase Identification** 
```
(A) python3

(R)

(N) 
```

**Response Suggestions**
```
(A) python3

(R)

(N) 
```

**Document Retrieval** 
```
(A) python3

(R)

(N) 
 
```

**Extractive Summarization** 
```
(A) python3

(R)

(N) 
```

**Interactive Mode**

TODO

### Reference
TODO