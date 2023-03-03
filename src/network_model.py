### creating the network model
## transforming the rules into network format
## creating the network
## running the classification algorithem
import networkx as nx
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from mlxtend.preprocessing import TransactionEncoder # add to requirements
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool

class rulesNetwork:

    def __init__(self, rules, genotype_data, train_x=None, test_x=None, train_y=None, test_y=None, transactions_columns = ["calls","status"], scoring_function=f1_score, base_model=DecisionTreeClassifier, 
                 test_size = 0.2, random_state = 10, encode_data = True):
        """A class to create a network out of the rules and build a classification model
        Parameters
        -----------
        rules : dataframe
            A dataframe of the rules
        
        genotype_data : dataframe
            A merged data frame of the genotype inference data

        transactions_columns : str/list (default ["calls","status"])
            The columns to use for transaction sets.

        scoring_function : function (default f1_score)
            The scoring metric for the network nodes and edges
        
        base_model : function (default DecisionTreeClassifier)
            The base model for classification. The model attrbutes the score for each node.

        test_size : int (default 0.2)
            The test split size
        
        random_state : int (default 10)
            Controls the shuffling applied to the data before applying the split. 
        """
        
        self.rules = rules
        self.genotype_data = genotype_data
        self.transactions_columns = transactions_columns
        self.scoring_function = scoring_function
        self.base_model = base_model
        self.test_size = test_size
        self.random_state = random_state
        self.encode_data = encode_data
        if any([x is None for x in [train_x, test_x, train_y, test_y]]):
            self.train_x, self.test_x, self.train_y, self.test_y = self.train_test_genotype()
        else:
            self.train_x, self.test_x, self.train_y, self.test_y = train_x, test_x, train_y, test_y#self.train_test_genotype()
            

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
        
        if type(self.transactions_columns) is list:
            dfs = []
            for col in self.transactions_columns:
                dfs.append(column_to_df(self.genotype_data, col))
            df = pd.concat(dfs, axis=1)
        else:
            df = column_to_df(self.genotype_data, self.transactions_columns)
        
        return df
            

    def train_test_genotype(self):

        if self.encode_data:
            dataset = self.encode_genotype()
        else:
            dataset = self.genotype_data

        X = dataset.drop(['status'], axis=1)
        y = dataset['status'].astype(float)

        train_x, test_x, train_y, test_y = train_test_split(X, y.to_numpy().reshape(-1,1), test_size=self.test_size, random_state=self.random_state)

        return train_x, test_x, train_y, test_y

    def rules_to_edges(self):
        """Converts the rules dataframe into a network edges
        Parameters
        -----------
        self.rules : dataframe
            A dataframe of the rules       
        Returns
        -----------
        A list of network edges
        """
        edges = []
        for ax,row in self.rules.iterrows():
            edges.append((row['antecedents'],row['consequents'],row['lift'],row['confidence'],row['antecedent support'],row['consequent support']))

        return edges
    
    def score_single_node(self, node):
        """Calcualtes the score for a single node
        Parameters
        -----------
        self.train_x : dataframe
            A dataframe of the training dataset
        self.test_x : dataframe
            A dataframe of the test dataset
        self.train_y : list
            A list of the train tags
        self.test_x : list
            A list of the test tags
        node : int
            index of the node to calculate the score for
        Returns
        -----------
        A score
        """
        # function that scores a single node
        X = self.train_x[node]
        clf = self.base_model().fit(X.to_numpy().reshape(-1,1),self.train_y)
        score = self.scoring_function(clf.predict(self.test_x[node].to_numpy().reshape(-1,1)),self.test_y)
        return score
    
    def score_multi_node(self, nodes):
        """Calcualtes the score for a multiple nodes
        Parameters
        -----------
        self.train_x : dataframe
            A dataframe of the training dataset
        self.test_x : dataframe
            A dataframe of the test dataset
        self.train_y : list
            A list of the train tags
        self.test_x : list
            A list of the test tags
        nodes : list
            indices of the node to calculate the score for
        Returns
        -----------
        A score
        """
        # function that scores a single node
        X = self.train_x[nodes]
        clf = self.base_model().fit(X,self.train_y)
        score = self.scoring_function(clf.predict(self.test_x[nodes]),self.test_y)
        return score
    
    def score_node_multiprocess(self, n, base_model, scoring_function, train_x, train_y, test_x, test_y, multi=False):
        """Calcualtes the score for a single node
        Parameters
        -----------
        self.train_x : dataframe
            A dataframe of the training dataset
        self.test_x : dataframe
            A dataframe of the test dataset
        self.train_y : list
            A list of the train tags
        self.test_x : list
            A list of the test tags
        node : int
            index of the node to calculate the score for
        Returns
        -----------
        A score
        """
        # function that scores a single node
        X = train_x[n]
        
        if multi:
            clf = base_model().fit(X,train_y)
            score = scoring_function(clf.predict(test_x[n]),test_y)
        else:
            clf = base_model().fit(X.to_numpy().reshape(-1,1),train_y)
            score = scoring_function(clf.predict(test_x[n].to_numpy().reshape(-1,1)),test_y)
        return score                                                


    def create_network(self):
        """Create the network from the rules dataframe
        Parameters
        -----------
        self.rules : dataframe
            A dataframe of the rules       
        Returns
        -----------
        A network graph
        """
        edges = self.rules_to_edges()
        graph = nx.DiGraph()
        # insert edges into graph
        for edge in tqdm(edges, desc="Creating network"):
            # add node goodness of fit using the scoring function as the node attribute  
            # get single feature column data
            a,b,lift,confidence,a_support,b_support = edge
            # A node score
            a_score = self.score_single_node(a)
            # B node score
            b_score = self.score_single_node(b)
            increase = self.score_multi_node([a,b])       
            graph.add_node(a,score=a_score, support=a_support)
            graph.add_node(b,score=b_score, support=b_support)
            graph.add_edge(a, b, increase_factor = (increase-a_score), # the edge value - the increase in the path score
                                 confidence = confidence, # the rule confidence value
                                 lift = lift, # the rule lift value
                                 lif = (increase-a_score)*lift, # a metric that consider the lift value
                                 cif = (increase-a_score)/(confidence+1e-6)) # a metric that consider the confidence value
        self.graph = graph
        self.max_score_node = max(nx.get_node_attributes(graph,'score'))

        return graph

    def create_network_multiprocess(self,processes=4):
            """Create the network from the rules dataframe
            Parameters
            -----------
            self.rules : dataframe
                A dataframe of the rules       
            Returns
            -----------
            A network graph
            """
            
            edges = self.rules[['antecedents','consequents','lift','confidence','antecedent support','consequent support']].copy() #self.rules_to_edges()

            nodes = list(set(edges.consequents))
            nodes.extend(x for x in edges.antecedents if x not in nodes)

            with Pool(processes) as p:
                args_generator = ((n, self.base_model, self.scoring_function, self.train_x, self.train_y, self.test_x, self.test_y, False) for n in nodes)
                results = p.starmap(self.score_node_multiprocess, args_generator)
                nodes_result_tuples = zip(nodes, results)
                single_node_score_dict = dict(nodes_result_tuples)
            
            edges['pair'] = edges[['antecedents','consequents']].apply(lambda x: ",".join(sorted(x)), axis = 1)

            pairs = list(set(edges['pair']))
            with Pool(processes) as p:
                args_generator = ((n.split(","), self.base_model, self.scoring_function, self.train_x, self.train_y, self.test_x, self.test_y, True) for n in pairs)
                results = p.starmap(self.score_node_multiprocess, args_generator)
                nodes_result_tuples = zip(pairs, results)
                multi_node_score_dict = dict(nodes_result_tuples)
            
            edges['a_score'] = edges['antecedents'].map(single_node_score_dict)
            edges['b_score'] = edges['consequents'].map(single_node_score_dict)
            edges['ab_score'] = edges['pair'].map(multi_node_score_dict)
            edges['increase_factor'] = edges['ab_score'] - edges['a_score']
            edges['lif'] = (edges['ab_score'] - edges['a_score'])*edges['lift']
            edges['cif'] = (edges['ab_score'] - edges['a_score'])/(edges['confidence'])+1e-6


            self.graph = nx.from_pandas_edgelist(
                    edges,
                    source = "antecedents",
                    target = "consequents",
                    edge_attr=["increase_factor", "lif","cif","lift"],
                    create_using=nx.DiGraph(),
            )

            nx.set_node_attributes(self.graph, single_node_score_dict, 'score')

            supp  = pd.Series(list(edges["antecedent support"]), index=edges.antecedents).to_dict()
            nx.set_node_attributes(self.graph, supp, 'support')
            
            self.max_score_node = max(nx.get_node_attributes(self.graph,'score'))

            return self.graph

    def get_max_increase_factor_edge(self, node, metric = 'increase_factor'):
        """Get the next node with the maximum increase in score
        Parameters
        -----------
        node : int
            the index of the next node to test

        metric : str (default increase_factor)
            which metric to use for the gready search. 
            increase_factor - the increase value
            confidence - the rule confidence value
            lift - the rule lift value
            lif - the increase factor divided by the lift value
            cif - the increase factor multiplied by the confidence
        
        self.graph : network
            the network graph

        Returns
        -----------
        A network graph
        """
        max_IF_node = None
        max_value = -1
        for edge in list(self.graph.out_edges(node)):
            IF = self.graph[edge[0]][edge[1]][metric]
            if IF > max_value:
                max_IF_node = edge[1]
                max_value = IF
        return max_IF_node, max_value

    def greedy_max_weight(self, metric = 'increase_factor', steps=10):
        """A greedy search to get the rules path from the network by taking the maximum value by metric
        Parameters
        -----------
        node : int
            the index of the next node to test

        metric : str (default increase_factor)
            which metric to use for the gready search. 
            increase_factor - the increase value
            confidence - the rule confidence value
            lift - the rule lift value
            lif - the increase factor divided by the lift value
            cif - the increase factor multiplied by the confidence
        
        steps : int (default 10)
            the number of maximum steps to take in the graph

        self.graph : network
            the network graph

        Returns
        -----------
        the rule path, the max path value, and the path values
        """
        current_node = self.max_score_node
        path =[current_node]
        path_energy_increase = [0]
        max_path_value = 0
        for _ in range(steps):
            # take a step in maximum IF direction
            next_step, step_IF = self.get_max_increase_factor_edge(current_node, metric)
            # accumalte IF
            max_path_value += step_IF
            # remmeber path
            path.append(next_step)
            # remmber change in max_path_value
            path_energy_increase.append(max_path_value)
            current_node = next_step

        return path,max_path_value,path_energy_increase
    
    def get_cliques_prediction(self, max_size = 10, weighted = False):
        if weighted:
            cliques = list(nx.algorithms.clique.max_weight_clique(nx.to_undirected(self.graph)))
        else:
            cliques = list(nx.algorithms.clique.find_cliques(nx.to_undirected(self.graph)))

        cl = min(cliques, key=lambda x:abs(len(x)-max_size))
       
        scores = self.get_prediction(cl)
        evaluate = self.evaluate_sg(cl)

        return cl, scores, evaluate

    def get_max_greedy_color(self):

        colors = nx.coloring.greedy_color(self.graph)

        vals = list(set(colors.values()))

        colors_groups = {}
        for pair in colors.items():
            if pair[1] not in colors_groups.keys():
                colors_groups[pair[1]] = []
            colors_groups[pair[1]].append(pair[0])
        
        max_val = max((len(v), k) for k,v in colors_groups.items())[1]
        
        return colors_groups[max_val]

    def get_prediction(self, path):
        """Get the prediction value based on the path
        Parameters
        -----------
        path : list
            A list of the nodes in the chosen path

        Returns
        -----------
        The prediction score value
        """
        clf = self.base_model()
        clf.fit(self.train_x[path], self.train_y)
        pred_y = clf.predict(self.test_x[path])
        value = self.scoring_function(self.test_y, pred_y)
        self.pred = value

        return value
    
    def get_confusion_matrix(self, path):
        """Get the confusion matrix based on the path
        Parameters
        -----------
        path : list
            A list of the nodes in the chosen path

        Returns
        -----------
        The confusion matrix
        """
        clf = self.base_model()
        clf.fit(self.train_x[path], self.train_y)
        pred_y = clf.predict(self.test_x[path])
        value = confusion_matrix(self.test_y, pred_y)
        self.confusion_matrix = value

        return value

    def edge_calc(self,edge):
        # add node goodness of fit using the scoring function as the node attribute  
        # get single feature column data
        a,b,lift,confidence = edge
        # A node score
        a_score = self.score_single_node(a)
        # B node score
        b_score = self.score_single_node(b)
        increase = self.score_multi_node([a,b])       
        self.graph.add_node(a,score=a_score)
        self.graph.add_node(b,score=b_score)
        self.graph.add_edge(a, b, increase_factor = (increase-a_score), # the edge value - the increase in the path score
                            confidence = confidence, # the rule confidence value
                            lift = lift, # the rule lift value
                            lif = (increase-a_score)*lift, # a metric that consider the lift value
                            cif = (increase-a_score)/(confidence+1e-6)) # a metric that consider the confidence value
    
    def evaluate_sg(self, clique, metric = "lift"):
        val = 0
        if metric=='lift':
            val = 1
        sg = nx.to_undirected(nx.subgraph(self.graph, clique))
        edges = sg.edges
        scores_edges = np.array([sg.get_edge_data(*edge)[metric] for edge in list(edges)])
        edge_count = sum(scores_edges>val)
        scores_edges = scores_edges[scores_edges>val]
        edge_sum = sum(abs(scores_edges))
        edge_sum = edge_sum/edge_count
        node_sum = 0
        for n in sg:
            node_sum += sg.nodes[n]['support']
        node_sum = node_sum/len(sg)
        return 1 - edge_sum + node_sum