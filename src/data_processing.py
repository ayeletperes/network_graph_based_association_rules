#### processing functions for genotype data
## reading genotype data and cleaning the dataset
## reading metadata
## creating a clean genotype table


## imports

import pandas as pd

# files import
from pathlib import Path
import glob
import re
from mlxtend.preprocessing import TransactionEncoder # add to requirements


class genotypeData:

    def __init__(self, path, metadata, file_pattern = "_genotype"):
        """A class to process genotype data created by either tigger of piglet. Currently only works for tigger bayesian inference
        Parameters
        -----------
        path : str
            A directory path for the genotype files

        metadata: dict
            A dict of the clinical status of each sample. Example: {'Sample1':'Healthy'}
            
        file_pattern : str (default _genotype)
            The pattern for the genotype files
        """
        self.path = path
        self.file_pattern = file_pattern
        self.metadata = metadata
        self.columns = set(["gene", "GENOTYPED_ALLELES"])

    def list_files(self):
        """listing all genotype tables in a directory
        Parameters
        -----------
        self.path : str
            A directory path for the genotype files
        
        self.file_pattern: str
            The pattern for the genotype files

        Returns
        -----------
        return a list of files
        """
        files = Path(self.path).glob('*'+self.file_pattern+'*')
        return files
    
    def read_genotype_data(self, file, sample, status):
        """reading genotype infernece table
        Parameters
        -----------
        file : str
            The path for the genotype file

        sample : str
            The genotype sample name

        status : str
            The clinical status of the sample

        Returns
        -----------
        A pandas dataframe of the genotyped data
        """
        data = pd.read_csv(file, sep="\t")
        ## check if columns exists; else exit with error
        if self.columns.issubset(set(data.columns)):
            data = data[["gene", "GENOTYPED_ALLELES"]]
            data = data.set_index("gene").apply(lambda x: x.str.split(',').explode()).reset_index()
            data['subject'] = sample
            data['status'] = str(status)
            return data
        else:
            raise ValueError(" ".join(self.columns)+" are required")
    
    def read_files(self, files):
        """Reading genotype files and merging tables
        Parameters
        -----------
        files : list
            A list of the genotype files found in the directory

        self.file_pattern: str
            The pattern for the genotype files

        self.metadata: dict
            A dict of the clinical status of each sample. Example: {'Sample1':'Healthy'}
        Returns
        -----------
        A merged dataframe of all the genotypes
        """
        dfl = list()
        for f in files:
            sample = re.sub(self.file_pattern, "", f.stem)
            status = str(self.metadata[sample])
            dfl.append(self.read_genotype_data(file = f, sample = sample, status = status))
        data = pd.concat(dfl, ignore_index=True)
        return data
   
    def clean_genotype_data(self, data, n_samples=None, silent=False):
        """Cleaning the genotype data by removing genes which are absent in indivduals and genes that only have one allele
        Parameters
        -----------
        data : dataframe
            a merged data frame of the genotype inference data
        n_samples : int (default None)
            the minimum required samples for gene to appear. If none, then gene must appear in all samples
        silent : bool (default False)
            If to suppress messages during cleanup
        Returns
        -----------
        A cleaned dataframe
        """

        sample_cut_off = len(set(data.subject))
        if n_samples is not None:
            sample_cut_off = n_samples
        
        # filter the dataset by genes and alleles thresholds
        n_alleles = len(data.groupby(['gene','GENOTYPED_ALLELES'])['subject'].nunique())
        data = data[data.groupby(['gene','GENOTYPED_ALLELES'])['subject'].transform('nunique') < sample_cut_off]

        if not silent:
            n_alleles2 = len(data.groupby(['gene','GENOTYPED_ALLELES'])['subject'].nunique())
            print('{} Alleles appeared in all {} individuals and were discarded.'.format(n_alleles-n_alleles2, sample_cut_off))
            
        n_genes = len(set(data.gene))
        data = data[data.groupby('gene')['subject'].transform('nunique') == sample_cut_off]

        if not silent:
            print('{} Genes did not appear in {} individuals and were discarded.'.format(n_genes-len(set(data.gene)), sample_cut_off))
        
        
        # add the calls column. 
        data["calls"] = data.apply(lambda x: x.gene + '*' + x.GENOTYPED_ALLELES, axis = 1)

        return data

    def column_to_df(self, data, col):
        """Pivots and encodes the dataframe to int
        Parameters
        -----------
        data : dataframe
            a merged data frame of the genotype inference data
        col : str
            the column to encode and pivot
        
        Returns
        -----------
        A pivoted dataframe
        """
        dataset = data.groupby('subject')[col]    
        index = list(dataset.groups.keys())
        dataset = dataset.apply(list).tolist()
        # encode the data
        te = TransactionEncoder()
        te_ary = te.fit(dataset).transform(dataset).astype('int')
        # create a matrix with the subject as index
        # if column is status leave just one column and rename
        if col=="status":
            df = pd.DataFrame(te_ary, columns=te.columns_, index=index)
            df.columns.values[0] = "status"
            df = df["status"]
        else:
            df = pd.DataFrame(te_ary, columns=te.columns_, index=index)
        
        return df

    def pivot_genotypes(self, data, columns = ["calls","status"]):
        """Pivots genotype dataframes given columns
        Parameters
        -----------
        genotype_data : dataframe
            a merged data frame of the genotype inference data
        columns : str/list
            the column to to pivot
        
        Returns
        -----------
        A pivoted dataframe
        """
        dfs = []
        for col in  columns:
            dfs.append(self.column_to_df(data, col))

        return pd.concat(dfs, axis=1)


        