import pandas as pd
import numpy as np
from itertools import combinations

# Define your classifiers and phenotypes
classifiers = ['JABS001', 'JABS002','JABS003','JABS004','JABS005','JABS006','JABS007','JABS008','JABS009']
phenotypes = ['duration_T5', 'duration_T20', 'duration_T55', 'numBouts_T5', 'numBouts_T20', 'numBouts_T55', 'avgBoutLen_T5', 'avgBoutLen_T20', 'avgBoutLen_T55']

# Generate synthetic genetic correlations for each pair of classifiers
for classifier1, classifier2 in combinations(classifiers, 2):
    # Create a DataFrame with random correlations between -1 and 1
    n_phenotypes = len(phenotypes)
    # Generate a symmetric matrix
    matrix_upper = np.triu(np.random.uniform(-1, 1, (n_phenotypes, n_phenotypes)))
    matrix_symm = matrix_upper + matrix_upper.T - np.diag(matrix_upper.diagonal())

    # Create a DataFrame from the matrix
    gen_corr_matrix = pd.DataFrame(matrix_symm, columns=phenotypes, index=phenotypes)
    # Set diagonal elements to 1
    np.fill_diagonal(matrix_symm, 1)
    
    # Save the DataFrame as a CSV file named as the pair of classifiers
    gen_corr_matrix.to_csv(f'gen_corr_{classifier1}_{classifier2}.csv')

