# Multi-labelling activities of retail stores from sales

We consider a retail store classification problem supported by a real word dataset. It includes yearly sales from several thousand stores with dirty categorical features on product labels and product hierarchies. Despite the fact that classification is a well-known problem of machine learning, current baseline methods are inefficient due to the dirty nature of data. As a consequence, We propose a practical solution using an intermediary global approximate classification of products, based on local products hierarchies.  Activities for a subset of stores are human-labeled to serve as ground truth for validation, and enable semi-supervision. These experiments show the effectiveness of our approach compared to baseline methods. 

This work is carried out thanks to the support of the [ANRT](https://www.anrt.asso.fr/fr) through the CIFRE (Industrial Agreements for Training through Research) program. The project is supported by [Bimedia](https://www.bimedia.com/) company and [LIAS laboratory (ISAE-ENSMA)](https://www.lias-lab.fr/).

## Repository Organisation

The key directories for experiments are organised as:

* **datasets**: Contains the Bimedia sales dataset and the validation dataset (with labelled stores) provided for the experiments.
* **thesaurusBT**: Contains all the files related to the experiments.
* **thesaurusBT/algortihms**: Contains the Python implementations of models used in our experiments or adaptation of implemented models that cannot be used as they are.
* **thesaurusBT/experiments/lib**: Contains the Python scripts where most of Python functions are defined.
* **thesaurusBT/experiments/preprocessing**: Contains the Python scripts and data ressources required for the execution of pre-processing algorithms.
* **thesaurusBT/experiments/processing**: Contains the Python scripts and data ressources required for the execution of processing algorithms.
* **thesaurusBT/experiments/results**: Contains the Python scripts for the evaluation and the reporting of workflows (pre-processing + processing) results.

## Requirements

The contribution is developped with the Python programming language. The minimal software requirements for the installation of this package are:

* Python 3
* PIP
* Git
* All operating systems that support Python

## Setup

At the root Git repository, execute the following command to install the **thesaurusBT** Python package:

```bash
pip install -e .
```

## Running

### A Complete Experimental Workflow

For complete experimental workflow execution, run **thesaurusBT** Python package. The arguments define the steps to include into the workflow during its execution. 

Available arguments are provided by the following command: 

```bash
python -m thesaurusBT -h
```

An example of the execution of a complete workflow of sales processing for the labeling of store activities:

```bash
python -m thesaurusBT -pre pivot -pro br_svm -cb off
```

In this example, the pre-precessing technic specified by the argument ``-pre`` is an aggregation pivot, the processing technic used for the multi-labelling is a Binary Relevance model with SVM as base classifier specified by the argument ``-pro`` and CatBOOST feature selection is not used as specified by the argument ``-cb``.

The results of the preprocessing and processing steps are stored in the **outputs** directory.

### All The Combinations Of Experimental Workflows

For the execution of all the combinations of experimental worflows, run **thesaurusBT** Python package with this command:

```bash
python -m thesaurusBT -all
```

The results of the preprocessing and processing steps are stored in the **outputs** directory.

## Results

In order calculate and report the scores archived by the experimentations, run **thesaurusBT** Python package with this command:

```bash
python -m thesaurusBT -res
```

This command display on the terminal the scores of each workflow exectuted, and create a file **results.csv**, saved in the **outputs** directory.

## Software license agreement

Details the license agreement of TME: [LICENSE](LICENSE)

## Historic Contributors (core developers first followed by alphabetical order)

* [Maxime PERROT (core developer)](https://www.lias-lab.fr/members/maximeperrot/) (Bimedia and LIAS/ISAE-ENSMA)
* [Mickael BARON](https://www.lias-lab.fr/members/mickaelbaron/) (LIAS/ISAE-ENSMA)
* [Brice CHARDIN](https://www.lias-lab.fr/members/bricechardin/) (LIAS/ISAE-ENSMA)
* [Stéphane JEAN](https://www.lias-lab.fr/members/stephanejean/) (LIAS/ISAE-ENSMA)
