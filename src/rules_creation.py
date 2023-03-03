### creating the rules
## tranformaing the dataframe into the rule type data
## running the rules algorithem
## creating the rules table

# data processing
import pandas as pd
import numpy as np

# pattern mining
from efficient_apriori import apriori # add to requirements
from mlxtend.preprocessing import TransactionEncoder # add to requirements
from mlxtend.frequent_patterns import fpgrowth # add to requirements
from mlxtend.frequent_patterns import association_rules # add to requirements

class rulesMining:

    def __init__(self, genotype_data, transactions_columns = "calls", encode_data = True):
        """A class that converts genotype data to rule mining data type and extract rules
        Parameters
        -----------
        genotype_data : dataframe
            A genotype data created with the genotypeData class
        transactions_columns : str/list (default calls)
            The columns to use for transaction sets.
        """
        
        self.genotype_data = genotype_data
        self.transactions_columns = transactions_columns
        self.encode_data = encode_data
    
    def encode_genotype(self):
        """Encodes the genotype table for desicion tree algorithem
        Parameters
        -----------
        self.genotype_data : dataframe
            A merged data frame of the genotype inference data
        Returns
        -----------
        A list of network edges
        """
        def column_to_df(genotype_data, col):
            dataset = genotype_data.groupby('subject')[col]    
            index = list(dataset.groups.keys())
            dataset = dataset.apply(list).tolist()
            # encode the data
            te = TransactionEncoder()
            te_ary = te.fit(dataset).transform(dataset)
            # create a matrix with the subject as index
            # if column is status leave just one column and rename
            df = pd.DataFrame(te_ary, columns=te.columns_, index=index)
            
            return df
        
        if type(self.transactions_columns) is list:
            dfs = []
            for col in self.transactions_columns:
                dfs.append(column_to_df(self.genotype_data, col))
            df = pd.concat(dfs, axis=1)
        else:
            df = column_to_df(self.genotype_data, self.transactions_columns)
        
        return df
    
    def genotypes_to_transactions(self):
        """Converts the genotypes to a sutiable dataframe for efficient_apriori rule mining
        Parameters
        -----------
        self.genotype_data : dataframe
            a merged data frame of the genotype inference data

        Returns
        -----------
        A nested list of transactions tuppels and subject keys.
        """

        # convert dataframe to list
        dataset = self.genotype_data.groupby('subject')[self.transactions_columns]
        index = list(dataset.groups.keys())
        dataset = dataset.apply(tuple).tolist()
        
        return index, dataset
    
    def mining_rules(self, algo='fpgrowth', min_support=0.0001, min_confidence=0.0001, max_len=2):
        """Rule mining using either fpgrowth or apriori algorithems.

        Parameters
        -----------
        algo : str (default: fpgrowth) 
            Which rule mining algorithem to use. 
            Either fpgrowth from mlxtend or apriori from efficient_apriori.

        min_support : int (default 0.0001)
            A float between 0 and 1 for minimum support of the itemsets returned.
            The support is computed as the fraction
            transactions_where_item(s)_occur / total_transactions.
        
        min_confidence : int (default 0.0001)
            A float between 0 and 1 for minimum confidence of the itemsets returned.
            The confidence is computed as the fraction
            support(antecedents-->consequents) / support(consequents).
    
        max_len : int (default 2)
            Maximum length of the itemsets generated. If `None` all
            possible itemsets lengths are evaluated.

        Returns
        -----------
        A dataframe of the association rules.
        """

        if algo not in ['apriori','fpgrowth']:
            raise ValueError("Selected algorithem is not supported. Please choose either apriori or fpgrowth")
        
        if algo=='fpgrowth':
            if self.encode_data:
                # encode the data
                df = self.encode_genotype()
            else:
                df = self.genotype_data

            # get frequent items
            frequent_itemsets = fpgrowth(df, min_support=min_support ,max_len=max_len, use_colnames=True)
            # get rules
            rules = association_rules(frequent_itemsets, min_threshold = min_confidence)
            if max_len > 2:
                rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)).astype("unicode")
                rules["consequents"] = rules["consequents"].apply(lambda x: list(x)).astype("unicode")
            else:
                rules["antecedents"] = rules["antecedents"].apply(lambda x: list(x)[0]).astype("unicode")
                rules["consequents"] = rules["consequents"].apply(lambda x: list(x)[0]).astype("unicode")

        else:
            # format data
            index, transactions = self.genotypes_to_transactions()
            # get frequenct items and rules
            frequent_itemsets, rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence, max_length=max_len)
            # convert the rules to dataframe
            attrs = [a for a in dir(rules[0]) if not a.startswith("_")]
            rules_rec = []
            for r in rules:
                rdict = {}
                for a in attrs:
                    rdict[a] = getattr(r, a)
                    rdict["rule"] = str(r).split("} (")[0]+"}"
                    rdict["len_l"] = len(r.lhs)
                    rdict["len_r"] = len(r.rhs)
                rules_rec.append(rdict)
            rules = pd.DataFrame(rdict)
            rules.set_index('rule', inplace=True)
            rules.rename(columns={'lhs': 'antecedents', 'rhs': 'consequents'}, inplace=True)
            rules = rules[['len_l', 'len_r', 'count_lhs', 'count_rhs','count_full',
                     'support', 'confidence', 'lift', 'rpf', 'conviction', 'antecedents', 'consequents']]

        return rules