# MHCLMDA
This study proposes a computational method based on multiple hypergraph contrastive learning (MHCLMDA) to predict miRNA-disease associations.


# Example
`To run MHCLMDA on your data, execute the following command from the project home directory:
'python main.py'.`

# Dependencies
```
MHCLMDA was implemented with python 3.8.15. To run MHCLMDA, you need these packages:    
torch               1.13.1+cu116
torch-cluster       1.6.0+pt113cu116
torch-geometric     2.2.0
torch-scatter       2.1.0+pt113cu116
torch-sparse        0.6.16+pt113cu116
torch-spline-conv   1.2.1+pt113cu116
numpy               1.23.5
scipy               1.8.1
scikit-learn        1.2.0
```

# Input
the input files include:
disease-gene associations, miRNA-gene associations, miRNA-disease associations, disease similarity data, miRNA similarity data.

# output
```
The AUC of the test data based on MHCLMDA.
```