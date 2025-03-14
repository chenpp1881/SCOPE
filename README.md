# SCOPE
SCOPE: Selection and Context-aware Optimized Prediction Engine for Code Classification Tasks

# Required Packages
- python 3+
- transformers 4.26.1
- pandas 1.5.3
- scikit-learn1.2.2
- pytorch 1.13.1

# Datasets
## Datasets for ESC and VSC
Our proposed method is empirically evaluated on seven benchmark datasets. I have summarized all the datasets at: [Dataset](https://zenodo.org/records/14017657)


# Running
To run program, please use this command: python `Main.py`.

Also all the hyper-parameters can be found in `Main.py`.

The index of the project is: ['Authorship', 'DefectPrediction', 'Java250', 'Python800', 'BigVul', 'Devign', 'Reveal']

Examples:

`
python Main.py --project_idx xx --model xx
`
