# drugFormulation-forecast

Reference journal / repo:  Towards safer and efficient formulations: Machine learning approaches to predict drug-excipient compatibility by Hang, N. et al (https://pubmed.ncbi.nlm.nih.gov/38341049/)

model_v1.py houses the model to train a one-to-one and one-to-many api-excipient pairs. The forecaster is trained in a supervised manner where the predictors are binary (0 = not compatible, 1 otherwise). 

data cleaning.ipynb houses the preprocessing for the dailymed_output_686.json file. Note that preprocecssing is not complete and still requires the following procedures:     
* Use UNII only data and run for proof of concept
* Use different source to identify UNII further 
* Normalize names if still issues
* 2D and mol2vec descriptors extraction
* Handle inbalancing (zero cases of false data)

lstm using journal data.ipynb houses the preprocessing for one-to-one data supplied by the reference repository. It's getting the dataset in the same format that the original authors used (creating  2D_data.csv & mol2vec_data.csv), which are fed into model_v1.py for initial screening. 

