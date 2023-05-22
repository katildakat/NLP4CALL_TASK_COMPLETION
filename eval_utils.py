import itertools
import numpy as np
import pandas as pd
import random
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import umap
import torch

def get_embed_dict(sentences, model):
    """
    Computes embeddings for a list of unique sentences using a pre-trained SentenceTransformer model.

    Parameters
    ----------
    sentences : list
        A list of sentences for which embeddings need to be computed.
    model : SentenceTransformer
        A SentenceTransformer model.

    Returns
    -------
    embed_dict : dict
        A dictionary containing the sentences as keys and their corresponding embeddings as values.
    """
    
    model.eval()
    sentences = list(set(sentences))
    embeddings = model.encode(sentences, batch_size=16, convert_to_numpy=True)
    embed_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
    
    return  embed_dict
    
def get_label_centroids(df, embed_column, label_column):
    """
    Computes the centroid of embeddings for each unique label in a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the embeddings and labels.
    embed_column : str
        The column name in the DataFrame where the embeddings are stored.
    label_column : str
        The column name in the DataFrame where the labels are stored.

    Returns
    -------
    centroid_dict : dict
        A dictionary containing the unique labels as keys and their corresponding centroid as values.
    """

    # get label centroids
    centroid_dict = {}
    for l in df[label_column].unique():
        label_df = df[df[label_column]==l].copy()
        label_embeddings = np.vstack(label_df[embed_column])
        label_centroid = np.array(label_embeddings).mean(axis=0)
        centroid_dict[l]=label_centroid
    
    return centroid_dict
    
def one_shot_cls(embed_column, train_df, test_df, label_column):
    """
    Performs one-shot classification on test data using pre-computed embeddings and label centroids.

    Parameters
    ----------
    embed_column : str
        The column name in the DataFrame where the embeddings are stored.
    train_df : pd.DataFrame
        DataFrame containing the training embeddings and labels.
    test_df : pd.DataFrame
        DataFrame containing the test embeddings and labels.
    label_column : str
        The column name in the DataFrame where the labels are stored.

    Returns
    -------
    y_preds : list
        A list of predicted labels for the test data.
    query_labels : list
        A list of true labels for the test data.
    true_dist_avg : float
        The average cosine distance between the test embeddings and their true label centroids.
    other_dist_avg : float
        The average cosine distance between the test embeddings and centroids of other labels.
    """
    
    
    y_preds = []
    true_dist_total = 0
    other_dist_total = 0
    
    centr_dict = get_label_centroids(train_df, embed_column, label_column)
    # embeds + true labels
    query_embeds = test_df[embed_column].tolist()
    query_labels = test_df[label_column].tolist()
    
    
    for i, q_embed in enumerate(query_embeds):
        q_label_true = query_labels[i] # get the true label of a query
        
        # compare a query to label centroids
        test_queries = [q_embed]*len(centr_dict) 
        cosine_distances = metrics.pairwise.paired_cosine_distances(test_queries, list(centr_dict.values()))
        
        y_pred_ind = np.argmin(cosine_distances) # get the ind of the closest centroid 
        y_pred = list(centr_dict.keys())[y_pred_ind] # get the label of the closest centroid 
        y_preds.append(y_pred)
        
        true_ind = list(centr_dict.keys()).index(q_label_true) # get the true centroid ind
        true_dist = cosine_distances[true_ind] # get the distance to the true centroid
        true_dist_total+=true_dist
        
        other_dist = 0 # average distance to other task centroids
        for j, d in enumerate(cosine_distances):
            if j != true_ind:
                other_dist+=d
        other_dist = other_dist/(len(cosine_distances)-1)
        other_dist_total+=other_dist
        
    return y_preds, query_labels, true_dist_total/len(y_preds), other_dist_total/len(y_preds)
    
def get_mean_score_by_fold(df, train_embed_column, test_embed_column, criterion, n=1):
    all_y_pred = []
    all_y_true = []
    sample_to_pred = {}
    
    for s in df['split'].unique():
        y_pred = []
        y_true = []
        train_df = df[df['split']!=s]
        test_df = df[df['split']==s]
        
        for i, row in test_df.iterrows():
            q_vec = row[test_embed_column]
            task = row["task"]
            task_df = train_df[train_df["task"]==task]
            task_vecs = task_df[train_embed_column].tolist()

            cosines = metrics.pairwise.cosine_similarity([q_vec], task_vecs)[0]
            vector_ids = np.argsort(cosines*-1) # inds of task responses sorted by their proximity
            closest_answers = vector_ids[:n] # inds of n closest responses
            values = task_df[criterion].values[closest_answers]
            mean_score = np.mean(values)
            y_pred.append(mean_score)
            y_true.append(row[criterion])
            
            sample_to_pred[row['sample']]=mean_score
       
        all_y_pred+=y_pred
        all_y_true+=y_true
        
    
    return all_y_pred, all_y_true, sample_to_pred
    
def get_bert_n_closest_score(train_df, test_df, embed_column, criterion, n=1):
    y_pred = []
    
    for i, row in test_df.iterrows():
        q_vec = row[embed_column]
        
        train_task_df = train_df[train_df['task']==row['task']]
        task_vecs = np.vstack(train_task_df[embed_column])
        
        distances = metrics.pairwise.paired_cosine_distances([q_vec]*len(task_vecs), task_vecs)
        vector_ids = np.argsort(distances) # inds of task responses sorted by their proximity
        closest_answers = vector_ids[:n] # inds of n closest responses
        values = train_task_df[criterion].values[closest_answers]
        mean_score = np.mean(values)
        y_pred.append(mean_score)
    
    return y_pred

def evaluate_cls(y_true, y_pred, ranges=None):
    if ranges!=None:
        y_true = get_hist_bin(y_true, ranges[0], ranges[1])
        y_pred = get_hist_bin(y_pred, ranges[0], ranges[1])
    else:
        y_true = [school_round(num) for num in y_true]
        y_pred = [school_round(num) for num in y_pred]
    bins = [x for x in range(min(y_true),max(y_true)+1)]
    print(metrics.classification_report(y_true, y_pred))
    cm, cm_norm = create_confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, cm_norm, bins)   

def create_confusion_matrix(y_true, y_pred, labels=None): 
    
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    
    summed_values = cm.sum(axis=1) # sum rows
    summed_values = summed_values[:, np.newaxis]
    normalized_matrix = cm/summed_values
    
    return cm, normalized_matrix


def plot_confusion_matrix(cm, normalized_matrix, classes, annotate=True):
    classes.sort()

    plt.figure(figsize = (10,8))
    colors = sns.light_palette((220, 50, 20), input="husl", n_colors=80)
    
    # cell values
    labels = (np.asarray(["{0:.1f}\n({1})".format(prcnt, value)
                      for prcnt, value in zip(normalized_matrix.flatten(),
                                              cm.flatten())])).reshape(cm.shape)
    
    
    if annotate:
        ax = sns.heatmap(np.around(normalized_matrix, 2),
                        annot=labels,
                        linewidths=.8,
                        fmt="",
                        cmap=colors)
    else:
        ax = sns.heatmap(np.around(normalized_matrix, 2),
                linewidths=.8,
                cmap=colors)
        
    
    ax.set(xticklabels=classes)
    ax.set(yticklabels=classes)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90) 
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.show()
      
def evaluate_reg(y_true, y_pred, title, ranges=None):
    if ranges!=None:
        y_true_cls = get_hist_bin(y_true, ranges[0], ranges[1])
        y_pred_cls = get_hist_bin(y_pred, ranges[0], ranges[1])
    else:
        y_true_cls = [school_round(num) for num in y_true]
        y_pred_cls = [school_round(num) for num in y_pred]
    print("MAE: ", metrics.mean_absolute_error(y_true, y_pred))
    print("MAE to cls: ", metrics.mean_absolute_error(y_true_cls, y_pred_cls))
    cor = stats.spearmanr(y_true, y_pred)
    if cor[1]<0.05:
        print("Correlation: ", cor[0])
        
    cor2 = stats.spearmanr(y_true_cls, y_pred_cls)
    if cor2[1]<0.05:
        print("Correlation cls: ", cor2[0])
        
            
    kappa = metrics.cohen_kappa_score(y_true_cls, y_pred_cls, weights='quadratic')
    print("Kappa:",kappa)
    
    plt.scatter(y_true, y_pred, s=10)
    plt.xlabel("true")
    plt.ylabel("predicted")
    plt.title(title)
    plt.show()
    
def school_round(num):
    if num - int(num) < 0.5:
        return int(num)
    else:
        return int(num) + 1
        
def get_hist_bin(values, range_min, range_max):
    n_bins = range_max-range_min+1
    
    bin_labels=[]
    
    _, bin_edges = np.histogram([x for x in range(range_min,range_max+1)], bins=n_bins)
    hist, bin_edges = np.histogram(values, bins=bin_edges)
    for v in values:
        i = 1
        while v>bin_edges[i]:
            i+=1
        b=i
        bin_labels.append(b)
    
    return bin_labels

def get_centroid_distances(centroid_dict):
    """
    Computes pairwise cosine distances between centroids in a dictionary.

    Parameters
    ----------
    centroid_dict : dict
        A dictionary containing labels as keys and their corresponding centroids as values.

    Returns
    -------
    distance_df : pd.DataFrame
        A DataFrame containing pairwise distances between centroids, with columns 'label_1', 'label_2', and 'distance'.
    """

    centoid_pairs = [p for p in itertools.combinations(centroid_dict.keys(), 2)]
    labels1 = [p[0] for p in centoid_pairs]
    embeds1 = [centroid_dict[l] for l in labels1]
    
    labels2 = [p[1] for p in centoid_pairs]
    embeds2 = [centroid_dict[l] for l in labels2]
    
    cosine_scores = metrics.pairwise.paired_cosine_distances(embeds1, embeds2)
    
    distance_df = pd.DataFrame()
    distance_df['label_1'] = labels1
    distance_df['label_2'] = labels2
    distance_df['distance'] = cosine_scores
    distance_df.sort_values(by=['distance'], ascending=False, inplace=True)
    
    return distance_df

def get_bin_centroid_distances(df, embed_column, bin_name):

    
    task_scores = {}
    for t in df['task'].unique():
        task_df = df[df['task']==t].copy()
        score_centroids = {}
        for b in task_df[bin_name].unique():
            bin_df = task_df[task_df[bin_name]==b].copy()
            bin_embeddings = np.vstack(bin_df[embed_column])
            bin_centroid = np.array(bin_embeddings).mean(axis=0)
            score_centroids[b] = bin_centroid
        task_scores[t]=get_centroid_distances(score_centroids)['distance'].mean()
    
    return task_scores

def get_cluster_score(df, label_column, embed_column):
    """
    Computes the Calinski-Harabasz cluster score for a DataFrame with pre-computed embeddings and labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the embeddings and labels.
    label_column : str
        The column name in the DataFrame where the labels are stored.
    embed_column : str
        The column name in the DataFrame where the embeddings are stored.

    Returns
    -------
    score : float
        The Calinski-Harabasz cluster score computed using the given DataFrame.
    """
    
    X = np.vstack(df[embed_column])
    score = metrics.calinski_harabasz_score(X, df[label_column].tolist())
    
    return score

def compute_task_scores(df, label_column, embed_column, give_out=False):
    """
    Computes and prints out task centroid distance and cluster score for a DataFrame with pre-computed embeddings.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the embeddings and labels.
    label_column : str
        The column name in the DataFrame where the labels are stored.
    embed_column : str
        The column name in the DataFrame where the embeddings are stored.
    give_out : bool, optional, default=False
        Whether to return the computed distance and cluster score.

    Returns
    -------
    distance : pd.DataFrame, optional
        A DataFrame containing the pairwise distances between task centroids.
    cluster_score : float, optional
        The cluster score computed using the given DataFrame.
    """

    centroids = get_label_centroids(df, embed_column, label_column)
    distance = get_centroid_distances(centroids)
    cluster_score = get_cluster_score(df, label_column, embed_column)
    
    if give_out:
        return distance, cluster_score
        
    else:
        print("TASK CENTROID DISTANCE: ", distance['distance'].mean())
        print("CLUSTER SCORE:", cluster_score)
    
def get_score_scores(df, embed_column, bin_name):
    task_scores = {}
    for t in df['task'].unique():
        task_df = df[df['task']==t].copy()
        task_scores[t] = get_cluster_score(task_df, bin_name, embed_column)
    
    return task_scores

def compute_bin_scores(df, embed_column, bin_name, give_out=False):
    score_centroids = get_bin_centroid_distances(df, embed_column, bin_name)
    score_scores = get_score_scores(df, embed_column, bin_name)
    
    print("BIN DISTANCES: ", np.mean(list(score_centroids.values())))
    print("BIN CLUSTER SCORE: ", np.mean(list(score_scores.values())))
    
    if give_out:
        return score_centroids, score_scores

def plot_n_random_tasks(df, task_column, embed_column, n=10):
    """
    Creates a 2D scatter plot of n randomly selected tasks from a DataFrame using UMAP for dimensionality reduction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the task information and embeddings.
    task_column : str
        The column name in the DataFrame where the tasks are stored.
    embed_column : str
        The column name in the DataFrame where the embeddings are stored.
    n : int, optional, default=10
        The number of random tasks to select and visualize.

    Returns
    -------
    None
    """
    
    random.seed(10)
    random_tasks = random.sample(list(df[task_column].unique()), n)
    print(random_tasks)
    num_tasks = len(random_tasks)
    
    umap_data = umap.UMAP(n_neighbors=15, 
                          n_components=2, 
                          min_dist=0.99, 
                          metric='cosine').fit_transform(np.vstack(df[df[task_column].isin(random_tasks)][embed_column]))
    

    
    # get discrete colormap
    if num_tasks < 21:
        cmap = plt.get_cmap('tab20b', num_tasks)
        colors = cmap.colors
    else:
        cmap = plt.get_cmap('terrain', num_tasks)
        colors = cmap(np.arange(0,cmap.N)) 

    c_dict={}
    for i, l in enumerate(random_tasks):
        c_dict[l]=colors[i]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    scatter = plt.scatter(umap_data[:,0], 
                          umap_data[:,1], 
                          c=[c_dict[t] for t in df[df[task_column].isin(random_tasks)][task_column]],
                          s=25)
    
    norm = mpl.colors.BoundaryNorm(np.arange(0, num_tasks+1), num_tasks)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ticks=np.arange(0.5, num_tasks, 1))
    cbar.set_ticklabels(list(c_dict.keys()))

    plt.show()
    
def plot_subtask(df, task_column, subtask, embed_column, score_column):
    
    task_df = df[df[task_column]==subtask].copy()

    umap_data = umap.UMAP(n_neighbors=3, 
                  n_components=2, 
                  min_dist=0.99, 
                  metric='cosine').fit_transform(np.vstack(task_df[embed_column]))
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    task_df = df[df[task_column]==subtask]
    
    scatter = plt.scatter(umap_data[:,0], umap_data[:,1], c=task_df[score_column], s=25)
    plt.colorbar()
    plt.show()
    
class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, transcripts):
        self.transcripts = transcripts

    def __len__(self):
        return len(self.transcripts)

    def __getitem__(self, transcript_id):
        item = {}
        item['transcripts'] = self.transcripts[transcript_id]

        return item
        
def get_mean_dict(sentences, model, tokenizer, device):
    model.to(device)
    model.eval()
    
    sentences = list(set(sentences))   
    sent_dataset = BERTDataset(sentences)
    data_generator = torch.utils.data.DataLoader(sent_dataset, batch_size=32, shuffle=False)
    
    MEANs = torch.tensor([]).to(device)
    
    for i, batch in enumerate(data_generator):
        encodings = tokenizer(batch['transcripts'], padding=True, return_tensors="pt" )
        if torch.cuda.is_available():
            encodings = encodings.to(device)
        with torch.no_grad():
            output = model(input_ids=encodings['input_ids'], attention_mask=encodings['attention_mask'])
        token_embeddings = output[0]
        input_mask_expanded = encodings['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        MEAN = torch.sum(token_embeddings*input_mask_expanded,1)/torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        MEANs = torch.cat((MEANs, MEAN), 0)
        
    embeddings = MEANs.detach().cpu().numpy()
    embed_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}   

    return embed_dict
    
def get_cls_dict(sentences, model, tokenizer, device):
    model.to(device)
    model.eval()
    
    sentences = list(set(sentences))   
    sent_dataset = BERTDataset(sentences)
    data_generator = torch.utils.data.DataLoader(sent_dataset, batch_size=32, shuffle=False)
    
    CLSs = torch.tensor([])
    for i, batch in enumerate(data_generator):
        encodings = tokenizer(batch['transcripts'], padding=True, return_tensors="pt" ).to(device)
        with torch.no_grad():
            CLS = model(input_ids=encodings['input_ids'], 
                        attention_mask=encodings['attention_mask'])['last_hidden_state'][:,0,:]
            CLS = CLS.to('cpu')
        CLSs = torch.cat((CLSs, CLS), 0)

    embeddings = CLSs.detach().numpy()
    embed_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}   
    
    return embed_dict