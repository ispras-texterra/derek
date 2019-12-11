![](https://github.com/ispras-texterra/derek/workflows/Tests/badge.svg)
## DEREK &ndash; Domain Entities and Relations Extraction Kit

### Goals and Tasks

This project's main focus is to provide semi-supervised models for solving information extraction tasks.
Given a collection of unlabeled texts and a set of texts labeled by entities and relations the tool is expected to extract corresponding entities and relations from any in-domain text automatically.
These tasks are critical for such learning facilitation activities like knowledge base construction, question-answering, etc. 

#### NER &ndash; Named Entity Recognition

This task's main concern is to find mentions of predefined types entities.
For example, sentence:

> ***Moscow*** is the major political, economic, cultural, and scientific center of ***Russia*** and ***Eastern Europe***, as well as the largest city entirely on the ***European continent***.

mentions 4 entities of type `Location`.

#### NET &ndash; Named Entity Typing

This task is focused around determining more detailed type of some typed entity.
For example, in sentence:

> ***Moscow*** is the capital and most populous city of ***Russia***

mention `Moscow` of type `Location` could be assigned a more fine grained type `City` or even `Capital`.

#### RelExt &ndash; Relation Extraction

This task is revolved around relations of several entities and their types.
For example, by considering a sentence:

> ***Daniel*** ruled ***Moscow*** as ***Grand Duke*** until 1303 and established it as a prosperous city.

we could determine that entities `Daniel` and `Moscow` of types `Person` and `Location` respectively are related with type `Leader of`.

### Technical Overview

#### Structure

DEREK project is written in Python and consists of:

1. `derek` runtime library which includes:
    1. data model (`data/model` module and some other modules in `data` package)
    1. [Tensorflow](https://www.tensorflow.org/) based implementations of neural networks for all tasks (`ner`, `net` and `rel_ext` packages)
    1. task-agnostic helpers code to simplify models development (`common` package)
    1. integrations with external preprocessors (tokenizers, POS taggers, syntactic parsers) such as [NLTK](https://www.nltk.org/), [UDPipe](http://ufal.mff.cuni.cz/udpipe), [TDozat](https://github.com/tdozat/Parser-v3) and internal ones such as Texterra and BabylonDigger (some modules in `data` package)
    1. readers for widely used corpora formats such as [BRAT](https://brat.nlplab.org/), [BioNLP](http://2016.bionlp-st.org/), [BioCreative](https://biocreative.bioinformatics.udel.edu/) (`data/readers` module)
1. unit and integration `tests` suite for runtime library
1. `tools` suite which consists of:
    1. preprocessing scripts (`generate_dataset`, `segment` and many other modules)
    1. evaluation scripts (`param_search`, `evaluate` and many other modules)
    1. simple HTTP `server` based on [Flask](http://flask.pocoo.org/) for model exploitation use cases
1. `derek-demo` application based on [MongoDB](https://www.mongodb.com/) containing:
    1. `processors` for processing crawled texts with some model (through a server)
    1. simple `ui` based on [Flask](http://flask.pocoo.org/) and [Jinja](http://jinja.pocoo.org/) for model results presentation and analysis

#### Abstractions

1. `Document` (`data/model`): tokenized text with some data attached to it:
    1. `TokenSpan`: span (range) of tokens
    1. `Sentence` (`Document.sentences`): `TokenSpan` describing sentence boundaries
    1. `Entity` (`Document.entities`): `TokenSpan` with a `type` attached describing entity mention
    1. `Relation` (`Document.relations`): pair of `Entity` instances with a `type` attached
    1. `Document.token_features`: each key in `dict` represents some additional information in form of a list with a value for each token
    1. `Document.extras`: each key in `dict` represents some additional information in a free form
1. `DocumentTransformer` (`data/transformers`): interface for classes which somehow transform `Document` to get another `Document`
1. `AbstractVectorizer` (`data/vectorizers`): parent for classes which somehow provide feature vectors (neural network driven transfer learning) for each token in `Document` (e.g., `FastTextVectorizer`)
1. FE &ndash; feature extractor: classes which extract features from objects (e.g., `Entity` in context of `Document`) in a format suitable for machine learning
1. graph factory, or simply graph: computational graph of a neural network which contains mathematical operations converting provided features into predictions, the most research-intensive part of the system
1. features meta: classes which contain meta information about features (e.g., dimensionality) extracted by some FE, they form bridges between FE and graph parts of algorithm 
1. classifier and trainer: classes which wire FE with graph factory and perform all required integration actions

### How to install DEREK library and tools?

To install DEREK library and tools you should:

1. clone this Git repository: `git clone ...`
1. `cd derek`
1. fetch submodules of the repo: `git submodule update --init --recursive`
1. install Python 3.6
1. install DEREK library dependencies: `pip3 install -r requirements.txt`
1. install DEREK tools dependencies: `pip3 install -r tools/requirements.txt`

### Pretrained models and resources

All images contains code, resources (if licence allows it) and training properties for results reproducing. Default entrypoint runs server for provided model, you only need to bind container's port 5000 to your local to get things done. 

#### CoNLL-03 NER

Reproducing [Ma and Hovy (2016)](https://arxiv.org/pdf/1603.01354.pdf) results. 

**DockerImage (on DockerHub) with best model** -- trifonovispras/derek-images:derek-ner-conll03

|                                        model                                       	| dev F1 	| test F1 	|
|:----------------------------------------------------------------------------------:	|:------:	|:-------:	|
|                                  DEREK (10 seeds)                                  	| 0.9488 	|  0.9101 	|
| SOTA without LM [[Wu et al., 2018](https://aclweb.org/anthology/D18-1310)] (5 seeds) 	| 0.9487 	|  0.9189 	|
| SOTA with LM [[Baevski et al., 2019](https://arxiv.org/pdf/1903.07785.pdf)] (1 seed) 	| 0.9690 	|  0.9350 	|

ImageNote: UDPipe `english-ud-2.0-170801.udpipe` model is used for segmentation on server (you can change it by storing other model and specifying other segmenter arguments when running an image).

#### Ontonotes v5 NER

Standard BiLSTM-CNN-CRF architecture with GloVe embeddings.

**DockerImage (on DockerHub) with best model** -- trifonovispras/derek-images:derek-ner-ontonotes

|                                             model                                            	| dev F1 	| test F1 	|
|:--------------------------------------------------------------------------------------------:	|:------:	|:-------:	|
|                                        DEREK (5 seeds)                                       	| 0.8679 	|  0.8506 	|
| SOTA without LM [[Ghaddar and Langlais, 2018](https://arxiv.org/pdf/1806.03489.pdf)] (5 seeds) 	| 0.8644 	|  0.8795 	|
|      SOTA with LM [[Akbik et al., 2018](https://aclweb.org/anthology/C18-1139)]  (2 seeds)     	|    -   	|  0.8971 	|

ImageNote: UDPipe `english-ud-2.0-170801.udpipe` model is used for segmentation on server (you can change it by storing other model and specifying other segmenter arguments when running an image).

#### FactRuEval-2016 NER

Standard BiLSTM-CNN-CRF architecture with word2vec embeddings.

**DockerImage (on DockerHub) with best model** -- trifonovispras/derek-images:derek-ner-factrueval

|                                                     model                                                     	|                    test F1                   	|
|:-------------------------------------------------------------------------------------------------------------:	|:--------------------------------------------:	|
|                                                DEREK (10 seeds)                                               	| 0.8114 (with LocOrg) 0.8464 (without LocOrg) 	|
| SOTA without LM [[FactRuEval 2016: Evaluation](http://www.dialog-21.ru/media/3430/starostinaetal.pdf)] (1 seed) 	|  0.809 (with LocOrg) 0.87 (without LocOrg)  	|

ImageNote: NLTK is used for segmentation on server (you can change it by storing other model and specifying other segmenter arguments when running an image).

#### ChemProt RelExt 

2-layered encoder model with postprocessed NLTK segmentation with/without SDP-multitask learning on GENIA corpus.

**DockerImage (on DockerHub) with best model** -- trifonovispras/derek-images:derek-relext-chemprot

|                                             model                                             	| dev F1 	| test F1 	|
|:---------------------------------------------------------------------------------------------:	|:------:	|:-------:	|
|                               DEREK without multitask (10 seeds)                              	| 0.6346 	|  0.6245 	|
|                                DEREK with multitask  (10 seeds)                               	| 0.6582 	|  0.6411 	|
| SOTA without LM [[Peng et al., 2018](https://arxiv.org/pdf/1802.01255.pdf)] (1 seed, train+dev) 	|    -   	|  0.6410 	|
|         SOTA with LM [[Peng et al., 2019](https://arxiv.org/pdf/1906.05474.pdf)] (1 seed)        	|    -   	|  0.7440 	|

ImageNote: NLTK with postprocessing is used for segmentation on server (you can change it by storing other model and specifying other segmenter arguments when running an image).

#### BB3 RelExt 

2-layered encoder model.

**DockerImage (on DockerHub) with best model** -- trifonovispras/derek-images:derek-relext-bb3

|                                                           model                                                          	| dev F1 	| test F1 	|
|:------------------------------------------------------------------------------------------------------------------------:	|:------:	|:-------:	|
|                                                     DEREK (20 seeds)                                                     	| 0.5662 	|  0.4942 	|
| SOTA without LM [[Li et al., 2017](https://www.inderscienceonline.com/doi/abs/10.1504/IJDMB.2017.085283)] (no information) 	|    -   	|  0.5810 	|

ImageNote: UDPipe `english-ud-2.0-170801.udpipe` is used for segmentation on server (you can change it by storing other model and specifying other segmenter arguments when running an image).

### How to train your own model and use it for prediction?

Training and employment of the model are series of the following steps:

1. `tools/generate_dataset.py` performs preparatory steps:
    1. It performs segmentation (sentence, token) of dataset if needed
    1. It preprocesses dataset with some heavyweight tools such as part-of-speech taggers, dependency parsers, etc. to ensure that these tools are applied to dataset just once
    1. It stores results in a pickle-based unified DEREK data representation
1. `tools/param_search.py` performs training of the model for different sets of hyperparameters, searches for the best performing model on development data and produces machine-/human-readable report of results
1. `tools/server.py` runs an HTTP service with a pretrained model inside and exposes a simple API for prediction

### Performing preparatory steps with `tools/generate_dataset.py`

1. setup correct Python module path: `export PYTHONPATH=$PWD:$PWD/babylondigger:$PWD/babylondigger/tdozat-parser-v3`
1. `python3 tools/generate_dataset.py -name train -input <where_is_train_data> -o <where_should_put_results> -transformer_props transformers.json <data_type> <segmenter_type_flags>`:
    1. `-name`, `-input` specify where input dataset is located. These flags could be specified multiple times if dataset is split to several parts, e.g, train/test or train/dev/test.
    1. `-o` specifies directory for output in a pickle-based unified DEREK data representation.
    1. `transformers.json` file is a specification of preprocessing steps. By default, you could simply put `{}` there. What you could specify here is discussed in details in the next section.
    1. `<data_type>` is type of dataset. Supported values are:
        1. `BioNLP`: BB3, SeeDev datasets
        1. `ChemProt`: CHEMPROT BioCreative dataset
        1. `BRAT`: BRAT annotation tool compatible dataset
        1. `FactRuEval`: FactRuEval-2016 Russian NER dataset
        1. `CoNLL`: CoNLL-03 European NER datasets
    1. `<segmenter_type_flags>` is type of segmenter and optional flags. Supported types of segmenters are:
        1. `nltk_segm` and `nltkpost_segm`: vanilla and postprocessed (fixes for typical mistakes) version of NLTK segmenter
        1. `texterra_segm -segm_key_path <key_file> {-segm_lang <lang>}`: Texterra API based segmenter for given API key (stored in file) and language
        1. `ud_segm <model_file>`: UDPipe based segmenter for given model

### What could be put in `transformers.json` for dataset preprocessing?

This file contains JSON object with a number of optional keys described in corresponding subsections.

#### Part-of-speech taggers / lemmatizers / dependency parsers

DEREK has integration with a number of morphological and syntactic tools: Texterra API, UDPipe, TDozat.

##### Texterra API

```json
{
    "tagger_parser": {"texterra": {"key_path": "<API_key_file>",
                                   "lang": "<manually specified language (optional)>"}}
}
```

##### UDPipe

```json
{
    "tagger_parser": {"babylondigger": [
        {"udpipe_tagger": {"model_path": "<model_file>"}},
        {"udpipe_parser": {"model_path": "<model_file>"}}
    ]}
}
```

##### TDozat

TDozat adapter additionally exposes a number of its internals (activations on different neural network layers) for transfer learning purposes.
How to employ them as features in your model will be discussed in one of the following sections.

Note that these vectors are quite large and could dramatically increase size of dataset unified representation.

```json
{
    "tagger_parser": {"babylondigger": [
        {"tdozat_tagger": {"model_path": "<model_file>",
                           "tagger_recur_layer": true,
                           "upos_hidden": true,
                           "xpos_hidden": true
        }},
        {"tdozat_parser": {"model_path": "<model_file>",
                           "parser_recur_layer": true
        }}
    ]}
}
```

##### Mixed UDPipe / TDozat

```json
{
    "tagger_parser": {"babylondigger": [
        {"udpipe_tagger": {"model_path": "<model_file>"}},
        {"tdozat_parser": {"model_path": "<model_file>"}}
    ]}
}
```

or vice versa

```json
{
    "tagger_parser": {"babylondigger": [
        {"tdozat_tagger": {"model_path": "<model_file>"}},
        {"udpipe_parser": {"model_path": "<model_file>"}}
    ]}
}
```

#### External Named Entity Recognition (NER)

##### Texterra API

Texterra API NER produces entities with quotation marks at both sides included.
If you want to exclude quotes then you could apply optional `remove_quotes` key. 

```json
{
    "ner": {"texterra": {"key_path": "<API_key_file>",
                         "lang": "<manually specified language (optional)>",
                         "remove_quotes": true}}
}
```

### Obtaining trained model and report of its behaviour with `tools/param_search.py` 

1. setup correct Python module path: `export PYTHONPATH=$PWD:$PWD/babylondigger:$PWD/babylondigger/tdozat-parser-v3`
1. `python3 tools/param_search.py -task <task name> -props <props.json> -lst <lst.json> -seeds <seeds> -out <where_should_put_results> <dataset_opts>`
    1. `<task name>` is one of the following: `ner` (Named Entity Recognition), `net` (Named Entity Typing), `rel_ext` (Relation Extraction)
    1. `<props.json>` and `<lst.json>` are JSON property files which contain values of all model hyperparameters.
    `<props.json>` contains base values of hyperparameters used throughout optimization experiments.
    `<lst.json>` has the same format but contains not single values but lists of values to search for the best performing model on each of them.
    If some property is present in both files its value from `<props.json>` is ignored.
    Concrete list of available properties is discussed in the following sections.
    1. `<seeds>` is a number of random seeds to evaluate each model on.
    Neural models behaviour and performance highly depends on random initialization, thus we recommend to employ at least 10 seeds for large datasets (e.g., CHEMPROT) and up to 50 seeds for smaller ones (e.g., BB3).
    Note that experiments are as fast as less seeds are used, thus there is always a trade-off between accuracy and efficiency.
    1. `<where_should_put_results>` is a directory where best performing models and extensive report of their behaviour for further analysis is placed.
    1. `<dataset_opts>` is a testing strategy: cross-validation (for datasets without established splits, dataset is split to given number of parts at runtime and model is evaluated on all folds) or holdout (for datasets with standard splits, these splits are constantly used for model evaluation).
        1. `cross_validation -traindev <dataset> -folds <folds>` where `<folds>` is a number of parts to split dataset on
        1. `holdout -train <train_dataset> -dev <dev_dataset>`

As a result `<where_should_put_results>` directory should have the following structure:

```
├── props_0
│   ├── split_0
│   │   ├── seed_0
│   │   │   ├── best_model_stats
│   │   │   │   ├── ... 
│   │   │   ├── best_model
│   │   │   │   ├── ... 
│   │   │   ├── best_results.json
│   │   ├── seed_1
│   │   │   ├── ...
│   │   ├── best_seed
│   │   ├── best_results.json
│   │   ├── mean_results.json
│   ├── split_1
│   │   ├── ...
│   ├── mean_results.json
│   ├── props.json
├── props_1
│   ├── ...
├── best_props
│   ├── ...
├── best_results.json
├── experiments_report.json
```

`param_search.py` iterates over all properties combinations specified via `<props.json>` and `<lst.json>`.
For every combination it evaluates model with given properties on all data splits. 
For `holdout` there is just one given split, for `cross_validation` there are `<folds>` splits generated on the fly by splitting given dataset.
On each split model is evaluated by running training procedure for a number of random seeds (to estimate stability of results).
Properties combinations are ranked by mean score over all splits and all seeds.

`best_results.json` contains all scores for the top 1 properties combination, its index is specified as `props_num` key.
Detailed results for this properties combination are provided in both `best_props` and `props_{props_num}` directories (their contents are identical).

Moreover, `experiments_report.json` provides detailed information about properties combinations ranking (top 5): common properties for best performers (`base_props`), their indices (`props_idx`), their properties difference (`difference`) and their scores (`scores`).
Detailed results for them are available in `props_{props_idx}` directories.

Each properties combination directory `props_N` contains: corresponding `props.json`, splits directories `split_N` and `mean_results.json` with mean scores over all splits.

Each split directory `split_N` contains: `mean_results.json` with mean scores over all seeds, `best_results.json` with the best performing seed scores (exposed as `seed_num`), seeds directories `seed_N` and `best_seed` directory (copy of `seed_{seed_num}`).

Each seed directory `seed_N` contains: `best_results.json` with the best performing epoch scores, `best_model_stats` with model predictions on dev dataset, `best_model` directory with the model itself for future usages.

### What could be put in `props.json` and `lst.json` for tuning model behaviour?

#### "Real world" examples

##### NER

NER model consists of one context encoder and classifier to label each token with respect to it's context.

`props.json` can be like:
```json
{
    "clip_norm": 5,
    
    "dropout": 0.5,
    "epoch": 30,
    
    "labelling_strategy": "BILOU",
    "learning_rate": 0.003,
    
    "loss": "crf",
    "encoding_size": 200,
    
    "ne_emb_size": 0,
    "optimizer": "adam",
    
    "pos_emb_size": 15
}
```

`lst.json` can be like:
```json
{
    "batcher": {"batch_size": [4, 8, 16]},
    "vectors_keys": [["fasttext"], ["fasttext", "upos_hidden"]]
}
```

##### NET

NET model consists of one context encoder and classifier to type each entity with respect to it's context.  

`props.json` can be like:
```json
{
    "batcher": {"batch_size": 4},
    "dropout": 0.5,
	"optimizer": "adam",
	"epoch": 10,
	"clip_norm": 5,
	"learning_rate": 0.005,
	"vectors_keys": ["fasttext"],
	"encoding_size": 200,
    
	"token_position_size": 20,
	"max_word_distance": 20,
	
	"types_to_collapse": ["PERSON"],
	
	"aggregation": {"attention": {}}
}
```

`lst.json` can be like:
```json
{
    "encoding_size": [100, 200]
}
```

##### RelExt

RelExt model architecture consists of two context encoders: shared one for multitask learning possibility and specific one for RelExt.
`"shared"` property specifies features for shared encoder, top-level properties -- for specific encoder.
In provided example we add shared encoding to specific features and pass result to specific encoder.

`props.json`:
```json
{
    "epoch": 20,
    "optimizer": "adam",
    "learning_rate": 0.005,
    "batcher": {"batch_size": 4},
    "clip_norm": 5,
    "dropout": 0.5,
    
    "max_candidate_distance": 20,
    
    "shared": {
       "models": [
            {
                "path": "glove.6B.100d.w2v.txt", 
                "trainable": false,
                "binary": false
            }
        ],
        "token_position_size": 15,
        "max_word_distance": 10
    },
    "encoding_size": 100,
    
    "add_shared": true,
    "entities_types_emb_size": 20,
    "specific_encoder_size": 100,
    
    "aggregation": {
        "take_spans": {}
    }
}
```

In `lst.json` we can tune parameters for nested dictionary properties.

`lst.json`:
```json
{
    "max_candidate_distance": [20, 25],
    
    "shared": {
        "token_log_position_size": [10, 20]
    }
}
```

#### Training procedure for all tasks

`epoch` is how many times to iterate training data on, too little leads to underfitting, too many leads to overfitting.
This is the maximal number of iterations, best model is chosen according to performance on dev data after each epoch <= `epoch`.
We recommend to analyze model convergence on some set of hyperparameters and one random seed to ensure that number of epoch is not too big as it makes evaluation of a model for each(!) set of hyperparameters longer and thus could slow down all experiments dramatically. 

Also it is possible to set number of `early_stopping_rounds` which defines how many epochs can be made without overall quality improvement on dev data.
It means that after another epoch a training procedure can be stopped if previous `early_stopping_rounds` epochs we didn't gain an improvement.     
For example, if `early_stopping_rounds: 0`, a training procedure will be stopped after first epoch without improvement on dev set.
We recommend to set this parameter in order not to train for maximum number of epochs and overfit on dev data. 

Learning rate is `learning_rate` / (1 + <num_of_completed_epochs> * `lr_decay`).

`clip_norm` is maximal norm of gradient for clipping.
Gradient clipping is important to reduce overfitting to outliers in dataset. 

Dropout is the most widely used regularizer for neural networks.
It randomly zeroes activations of layers during training to prevent feature co-adaptation.
For simplicity DEREK models apply dropout before each layer with the same zeroing probability: 1 - `dropout`.

```json
{
    "epoch": 10,
    "early_stopping_rounds": 3,
    "optimizer": "adam | adagrad | momentum | adadelta | rmsprop | nadam | sgd",
    "learning_rate": 0.01,
    "lr_decay": 0,
    "batcher": {"batch_size": 8},
    "clip_norm": 1,
    "dropout": 0.8
}
```

#### Word embeddings features properties

Several word embedding models could be used simultaneously: e.g., general model with a large dictionary and domain-specific model with smaller dictionary but better in-domain similarity.

Currently word2vec, fastText and TDozat word embedding models are supported.
GloVe models could be preliminarily converted to word2vec ones via `gensim` tool.

##### word2vec

Both text and binary formats are supported: could be specified via `binary` flag.

All models could be fine-tuned on target task (see `trainable` flag) but we do not recommend doing this for small datasets (e.g., BB3).

Several dimensionality reduction heuristics could be applied: lowercasing words (`lower`), replacing all digit to 0 (`replace_digits`), replacing all quotation marks (`replace_quotes`).
Note that all heuristics could be applied if and only if same preprocessing was applied before word embeddings model training. 

```json
{"models": [
    {"path": "<first_word2vec_file>", "binary": true, "trainable": false,
     "lower": false, "replace_digits": false, "replace_quotes": false},
    {"path": "<second_word2vec_file>"}
]}
```

##### fastText

Note that fastText models could not be fine-tuned as unlike word2vec they have open vocabulary.
The same preprocessing heuristics are available for fastText models as for word2vec ones.

```json
{
    "transformers": {
        "vectorizers": {
            "fasttext": [
                {"path": "<first_fasttext_file>",
                 "lower": false, "replace_digits": false, "replace_quotes": false},
                {"path": "<second_fasttext_file>"}
            ]
        }
    },
    "vectors_keys": ["fasttext"]
}
```

##### TDozat tagger

These features require turning tagger on and setting it for corresponding layer activations exposure at `generate_dataset.py` stage (see above sections).

```json
{
    "vectors_keys": ["tagger_recur_layer", "upos_hidden", "xpos_hidden"]
}
```

##### TDozat parser

These features require turning parser on and setting it for corresponding layer activations exposure at `generate_dataset.py` stage (see above sections).

```json
{
    "vectors_keys": ["parser_recur_layer"]
}
```

#### Character convolution features properties

Many NLP tasks require looking at text on subword level.
Simple and efficient way of doing that is applying CNN on characters: we first project each character into an low-dimenisional embedding, then apply convolution layer which models ngrams, then perform max pooling to select most informative character features.

In the given example we use 20 as dimensionality for characters, we apply three groups of convolutions (unigrams, bigrams, trigrams) which are projected to vectors of sizes 20, 30, 40 respectively to finally give 90 features per word.

```json
{
    "char_embedding_size": 20,
    "char_kernel_sizes": [1, 2, 3],
    "char_kernel_num_features": [20, 30, 40]
}
```

#### Morphological features properties

These features require turning tagger on at `generate_dataset.py` stage (see above sections).

For all features zero size means "one-hot", positive size means "projection to embedding of given size".
In the given example we encode part-of-speech tags as one-hot and project gender, animacy and number grammemes to the embedding space of size 10.

If you are not familiar with grammemes you could either omit `morph_feats_list` property (DEREK will use default list) or refer to Universal Dependencies documentation for the complete list and details. 

```json
{
    "pos_emb_size": 0,
    "morph_feats_emb_size": 10,
    "morph_feats_list": ["Gender", "Animacy", "Number"]
}
```

#### Syntactic features properties

These features require turning parser on at `generate_dataset.py` stage (see above sections).

`dt_label` is a label on an syntactic edge from word to its parent.
`dt_distance` is an offset (in words) of word with respect to its parent, offset modulo is clipped by `max_dt_distance`.
`dt_depth` is a depth of word in syntactic tree, depth is clipped by `max_dt_depth`.
For all features zero size means "one-hot", positive size means "projection to embedding of given size".

```json
{
  "dt_label_emb_size": 0,
  "dt_distance_emb_size": 10, 
  "max_dt_distance": 5,
  "dt_depth_emb_size": 10,
  "max_dt_depth": 5
}
```

#### Gazetteer features properties

Sometimes in NLP tasks you can provide additional useful features by marking some predefined specific tokens.
This is known as gazetteer features. You can prepare txt-file with such tokens on separated lines and turn on this feature.
Also you can specify to lowercase and/or lemmatize tokens before checking them in gazetteer.
Note: gazetter is not lemmatized even if this feature is on. Only document tokens are affected, so prepare lemmatized gazetteer if you want to use them (you can use `tools/gazetteer_lemmatizer.py` module).

```json
{
  "gazetteers": [
    {"path": "tests/data/feature_extractor/gazetteer.txt", "lower": false, "lemmatize": true}
  ]
}
```


#### External NER features properties (NER task only)

These features require turning external NER on at `generate_dataset.py` stage (see above sections).

Zero size means "one-hot", positive size means "projection to embedding of given size".

External NER results are encoded via `ne_labelling_strategy`: could be `IO` (inside-outside, default), `BIO` (begin-inside-outside), `BIO2` (begin-inside-outside but begin is applied just in ambiguous cases), `BILOU` (begin-inside-last-outside-unit).

```json
{
  "ne_emb_size": 10,
  "ne_labelling_strategy": "IO"
}
```

#### Position(s) wrt to entity(ies) features properties (NET and RelExt tasks only)

In NET task we classify an entity and therefore each token in its context has some kind of position wrt entity.
In RelExt task we classify pair of entities and thus each token in their context has positions wrt each of entities.

Positions could be computed in very different ways.
`token_position` is an offset (in words) of word with respect to entity, offset modulo is clipped by `max_word_distance`, such feature is often referred to in literature as "Word Position Embedding" (WPE).
`token_log_position` is similar to the previous way but logarithm of offset is used instead of offset itself.
`at_root_dt_path` is a flag of word being on path in syntactic tree from root word to entity.
For all features zero size means "one-hot", positive size means "projection to embedding of given size".

```json
{
    "token_position_size": 25,
    "token_log_position_size": 15,
    "max_word_distance": 8,
    "at_root_dt_path_size": 0
}
```

#### Context encoder properties

Above we described how turn on and off features fed in neural network for different tasks.
These features form a feature vector for each word in text.
Most NLP tasks are context-sensitive (making decision for a word requires looking at its neighbours).
Thus we should somehow encode context via words' feature vectors.
Typically three architectures are used to do this: BiLSTM (default), CNN and Transformer.

All encoder architectures share common properties: encoding size (dimensionality of output vectors) and skip connection (whether input feature vectors should be concatenated with output ones; default is not).

```json
{
    "encoding_size": 100,
    "skip_connection": false
}
```

##### BiLSTM / GRU

BiLSTM and its simplified version GRU are memory-driven, they consume word after word and "memorize" patterns.
Being a serious advantage (humans also read text sequentially) it is also a disadvantage: they are linear and could not work in parallel.

```json
{
    "encoding_type": "lstm | gru"
}
```

##### CNN

CNN looks just at local (ngram or kernel) context and therefore is the most computationally efficient (ngrams are linear and could be processed in parallel) but "stupid" approach.

```json
{
    "encoding_type": "cnn",
    "encoder_kernel_size": 3
}
```

##### Transformer

Transformer is about word-to-word interactions (self-attention mechanism).
Here we have quadratic algorithm but computations can be processed in parallel (for each pair of words).
It is very efficient while running on GPUs, at the same time on CPUs there is no enough parallelism to compensate non-linearity.
It is recommended in literature to apply several transformer layers to achieve generalization ability as of BiLSTM.

```json
{
  "encoding_type": "transformer",
  "encoder_layers_num": 6
}
```

#### Context aggregation (NET and RelExt tasks)

After words with their surrounding context are encoded into feature vectors we should somehow aggregate them to perform final classification.
There are several ways to do this: e.g., we could apply max or mean pooling (choose max value / compute average of each feature for the whole word sequence).
Alternatively, we could employ attention mechanism: this is similar to mean pooling but each word has its own different weight.
Finally, we could use context encoding feature vectors for some words directly: either first/last token indices of entity(ies) classified or first/last tokens of the whole word sequence.

Note that DEREK models could apply several aggregations simultaneously: in some cases it improves model performance.

##### Mean / max pooling

You could optionally apply non-linear dense layer before pooling to transform context encoding vector space and change its dimensionality.
To do this specify `dense_size` property with expected size of features vectors to perform pooling on.

```json
{
    "aggregation": {
        "max_pooling": {
            "dense_size": 100
        },
        "mean_pooling": {
            "dense_size": 100
        }
    }
}
```

##### Bahdanau-like attention

Similarly to pooling you could optionally apply non-linear dense layers before attention to transform context encoding vector space and change its dimensionality.
First, you could optionally project context encoding to `dense_size`.
Obtained vectors will be used for both attention weights and weighted sum computations.
Further, you could optionally project vectors of size `dense_size` or original context encoding (if no `dense_size` specified) to `non_linearity_size`.
In this case these vectors will instead be employed for attention weights computations.

By default, there is just one attention weight for each word in sequence.
Enabling `attention_per_feature` flag allows to have as many attention weights as `dense_size` (if given) or size of original context encoding.
This gives model additional flexibility to attend on different features differently but makes model having more parameters. 

```json
{
    "aggregation": {
        "attention": {
            "dense_size": 100,
            "non_linearity_size": 100,
            "attention_per_feature": false
        }
    }
}
```

##### Indexed aggregation: entity(ies) / sequence boundaries

Both aggregations are applicable only for BiLSTM context encoder as they use forward and backward encodings separately.

`take_spans` employs forward encoding for the last token(s) of classified entity(ies) and backward encoding of the first token.
This way model is expected to "memorize" information about both left/right context and entity(ies) it(them)self(ves).

`last_hiddens` is similar to `take_spans` but uses the first / last token of the whole sequence.

```json
{
    "aggregation": {
        "take_spans": {},
        "last_hiddens": {}
    }
}
```

#### NER-specific properties

NER model performs sequence labeling of words wrt their context: either via independent classification (`cross_entropy` `loss`) or Conditional Random Fields (`crf` `loss`).

This requires encoding of tokens wrt entities at train phase and decoding of entities wrt to predictions at test phase.
Strategy for encoding/decoding could be specified via `labelling_strategy`: could be `IO` (inside-outside), `BIO` (begin-inside-outside, default), `BIO2` (begin-inside-outside but begin is applied just in ambiguous cases), `BILOU` (begin-inside-last-outside-unit).

```json
{
    "labelling_strategy": "BIO",
    "loss": "cross_entropy | crf"
}
```

#### NET-specific properties

NET model classifies entities to assign more detailed types to them.

Original type of entity could be fed at both attention and classifier layers to adjust predictions.
Zero size means "one-hot", positive size means "projection to embedding of given size".

```json
{
    "ne_type_in_attention_size": 10,
    "ne_type_in_classifier_size": 0 
}
```

#### RelExt-specific properties

RelExt model classifies pairs of entities to assign relation type to them.

Pairs with too large token distance between entities (larger than `max_candidate_distance`) are filtered out on training and always predicted as having no relation.

Original types of entities (either jointly `rel_args` or independently `entities_types`) could be fed at both attention and classifier layers to adjust predictions.
Besides that, distances (original or logarithmic, clipped by `max_token_entities_distance`) between entities in pair and relation direction (is first entity at left / inside / at right of second entity?) could be employed. 
For all features zero size means "one-hot", positive size means "projection to embedding of given size".

```json
{
    "max_candidate_distance": 20,
    "rel_args_in_attention_size": 25,
    "rel_args_in_classifier_size": 0,
    "entities_types_in_attention_size": 25,
    "entities_types_in_classifier_size": 0,
    
    "entities_token_distance_in_attention_size": 25,
    "entities_token_log_distance_in_attention_size": 25,
    "entities_token_distance_in_classifier_size": 0,
    "entities_token_log_distance_in_classifier_size": 0,
    "max_token_entities_distance": 10,
    
    "rel_dir_in_attention_size": 0,
    "rel_dir_in_classifier_size": 0
}
```

Also RelExt model can be improved with shortest dependency path (SDP) multi-task learning. 
In this case model learns to predict SDP between two tokens during training time besides relation extraction.
To use this feature you should run `tools/param_search.py` with flag `-unlabeled <path_to_conllu_file>` and provide dataset file in CoNLL-U format containing dependency trees and POS tags.

Model architecture contains 2 encoders: 1 shared for multi-task and 1 specific for RelExt. You should specify shared-between-tasks features in `shared` nested properties dictionary and RelExt-specific features in top-level dictionary.

Encoders properties should be specified in top-level dictionary. 
To set settings for specific one you should add prefix `specific_encoder` to encoder features names: `specific_encoder_type`, `specific_encoder_size`, `specific_encoder_skip_connection` etc.  

`sdp_config` nested dictionary is used to set settings for SDP predicting layer. You need to specify all default training properties from `Training procedure for all tasks` chapter.

Also you can specify what kind of token pairs to take for SDP prediction by using `POS_types` property (default is all noun pairs in PTB format):
```json
{
    "POS_types": [["NN", "NN"], ["NNS", "NNS"]]
}
```

Ratio of SDP task training samples to RelExt ones is controlled by `sdp_samples_ratio`.

Overally, props architecture is kinda like:
```json
{
    "rel_ext_training_prop_1": 0,
    "rel_ext_training_prop_n": 0.05,
    
    "specific_encoder_prop_1": 100,
    "specific_encoder_prop_k": "lstm",
    
    "common_encoder_prop_1": 100,
    "common_encoder_prop_n": "gru",
    
    "shared": {
      "common_feature_prop_1": [],
      "common_feature_prop_k": 20
    },
    
    "sdp_config": {
      "sdp_training_prop_1": 0,
      "sdp_training_prop_n": 1
    },
    
    "sdp_samples_ratio": 1
}  
```

Example:
```json
{
    "optimizer": "adam",
    "learning_rate": 0.005,
    "batcher": {"batch_size": 4},
    "dropout": 0.5,
    "clip_norm": 5,
    "epoch": 50,
    "max_candidate_distance": 20,
    "entities_types_emb_size": 20,
    "aggregation": {
        "take_spans": {}
    },
     
    "add_shared": true,
    "specific_encoder_size": 100,
    
    "encoding_size": 100,

    "shared": {
        "models": [
            {
                "path": "emb_model.txt", 
                "trainable": false, 
                "binary": false
            }
        ],
        "token_position_size": 15,
        "max_word_distance": 10,
        "pos_emb_size": 15,
        "dt_label_emb_size": 15
    },
    
    "sdp_config" : { 
        "learning_rate": 0.005, 
        "loss": "cross_entropy", 
        "dropout": 0.5, 
        "clip_norm": 1, 
        "batcher": {"batch_size": 16}
    },

    "sdp_samples_ratio": 1
}
```

Also you can discard `sdp_config`, `sdp_samples_ratio` and CoNLL-U dataset to use this model without multi-task.

### Performing predictions via trained model with `tools/server.py`

#### How to run HTTP server for trained model?

1. setup correct Python module path: `export PYTHONPATH=$PWD:$PWD/babylondigger:$PWD/babylondigger/tdozat-parser-v3`
1. `python3 tools/server.py {-remote} {-port <port number>} {-ner <model_file>} {-rel_ext <model_file>} {-transformer_props transformers.json} <segmenter_type_flags>`:
    1. `-remote` if you want to access server from network, not just from localhost
    1. `-port <port number>` if you want to expose server under port different to default 5000
    1. `-ner <model_file>` to specify NER model to use, without NER model server clients will be required to always detect entities themselves
    1. `-rel_ext <model_file>` to specify RelExt model to use, without RelExt model server clients will not be able to request relations
    1. `-transformer_props transformers.json` specifies preprocessing pipeline required by NER/RelExt models, in most cases it is expected that file used at `generate_dataset` stage is reused here as is for consistency
    1. `<segmenter_type_flags>` are described in details in `generate_dataset` section, in most cases it is expected that arguments used at `generate_dataset` stage are reused here as they are for consistency

#### What is an API exposed by HTTP server?

Server exposes the only API route `/` with two optional query parameters `entities=1` and `relations=1` to request entities and relations for given document respectively.

Document is fed in the following JSON format:

```json
{
    "text": "Ivan meets Vladislav at work.",
    "entities": [
        {"id":  "T1", "start": 0, "end": 4, "type": "Person"},
        {"id":  "T2", "start": 14, "end": 23, "type": "Person"}
    ]
}
```

`entities` field is optional and means that given entities will be employed instead of ones determined by NER model.

Server provides client with document in a similar JSON format:

```json
{
    "entities": [
        {"id":  "T1", "start": 0, "end": 4, "type": "Person"},
        {"id":  "T2", "start": 14, "end": 23, "type": "Person"}
    ],
    "relations": [
        {"first": "T1", "second": "T2", "type": "Knows"}
    ]
}
```

Presence of `entities` and `relations` keys is controlled via respective query parameters (see above).

#### How to prepare Docker container with HTTP server, trained model and required resources?

1. Prepare `model` folder in source path with trained model (must contain `extractor`, `model`, `graph.pkl` and other files) or several folders for several models
1. Prepare `resources` folder in source path with preprocessing models and `transformers.json` containing relative paths(for example, `resources/udpipe_model.udpipe`)
1. Prepare DockerFile in source path using provided template; you must configure ENTRYPOINT to run server.py (see chapter `How to run HTTP server for trained model`).
 : 
     ```dockerfile
    FROM ubuntu:18.04
    
    RUN apt-get update \
        && DEBIAN_FRONTEND=noninteractive apt-get install -y locales \
        && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
        && dpkg-reconfigure --frontend=noninteractive locales \
        && update-locale LANG=en_US.UTF-8
    ENV LANG en_US.UTF-8
    ENV LANGUAGE en_US:en  
    ENV LC_ALL en_US.UTF-8
    
    RUN apt-get update && apt-get install -qq -y python3 python3-pip
    COPY requirements.txt /derek/
    RUN pip3 install -r /derek/requirements.txt
    RUN python3 -c "import nltk;nltk.download('punkt')"
    COPY babylondigger /derek/babylondigger
    COPY tools /derek/tools
    RUN pip3 install -r /derek/tools/requirements.txt
    COPY derek /derek/derek
    
    COPY model /derek/model
    COPY resources /derek/resources
    
    WORKDIR derek
    ENV PYTHONPATH .:babylondigger:babylondigger/tdozat-parser-v3
    ENTRYPOINT ["python3", "tools/server.py", "-remote", "-ner", "<model_file>", "-rel_ext", "<model_file>", "-transformer_props", "resources/transformers.json", "<segmenter_type_flags>"]
    ```
1. Build image with command `docker build . -t derek-container-name`
1. Run container `docker run -p 80:5000 -d derek-container-name`. This command binds host's port 80 to container's port 5000, change it if you wish.
1. Now you can send requests for server available on 80'th port. 