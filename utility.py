import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def combine_data(train, test):
    '''
    Combines the two given dataframes to one. Adds a column
    'TrainTest' that indicates the origin of the data row.
    '''
    train["TrainTest"] = "Train"
    test["TrainTest"] = "Test"
    return pd.concat([train, test], sort=False)

def split_data(data):
    '''
    Splits the given dataframe into training and testing data.
    Removes the column 'TrainTest'.
    '''
    train = data[data["TrainTest"]=="Train"].drop(labels="TrainTest", axis=1)
    test = data[data["TrainTest"]=="Test"].drop(labels=["TrainTest", "Survived"], axis=1)
    return train, test

def plot_hist(df, by, bins=10, show_median_mean=True, **kwds):
    '''
    Plots a histogram with vertical lines indicating median and mean values.
    
    Parameters
    ----------
    df : pandas DataFrame.
    by : str
        column for the histogram.
    bins : int, default 10 
        Number of histogram bins to be used.
    show_median_mean : bool, default True
        If True, vertical lines indicating median and mean values
        will be added to the plot.
    **kwds
        Additionaal keyword arguments that will be passed to
        pd.DataFrame.plot.hist
            
    Returns
    -------
    axes : matplotlib.AxesSubplot histogram.
    
    '''
    fig, ax = plt.subplots()
    ax = df[by].plot.hist(bins=bins, ax=ax, color="slategray", label="_nolegend_", **kwds)
    ax.set_title(f"Distribution of {by}")

    if show_median_mean:
        mean = df[by].mean()
        median = df[by].median()
        ax.axvline(mean, color="red", ls="--", label=f"mean = {mean:.2f}")
        ax.axvline(median, color="orange", ls="-.", label=f"median = {median:.2f}")
        ax.legend()
    
    ax.set_title(f"Distribution of {by}")
    
    return ax

def plot_cos(df, feature, add_labels=True, figsize=(10,5)):
    '''
    Plots the Chance of Survival (cos) over a given feature.
    
    Parameters
    ----------
    df : pandas DataFrame.
    feature : str
        The column to plot the CoS for.
    add_labels : bool
        If True, data labels will be added.
    figsize : tuple of ints
        The size of the plot.
        
    Returns
    -------
    axes : matplotlib.AxesSubplot bar chart.
    
    '''
    try:
        df_plot = df[df["TrainTest"]=="Train"]
    except:
        df_plot = df.copy()
    df_plot = df_plot.groupby([feature, "Survived"]).size().reset_index()
    df_plot = df_plot.pivot(columns=feature, index="Survived", values = 0)
    df_plot = df_plot.apply(lambda x: x/x.sum()).sort_values(by="Survived", ascending=False)
    ax = df_plot.transpose().plot(kind="bar", stacked=True, color=["green", "lightgray"],
                                  figsize=figsize)
    ax.set_title(f"Chance of Survival based on {feature}")
    ax.get_legend().remove()
    
    if add_labels:
        vals = df_plot.values
        yoff = 0.05
        for col in range(len(vals[0])):
            val = vals[0][col]
            if pd.isna(val):
                val = 0
            label = "{:,.0%}".format(val)
            xpos = col
            if val > 0.9:
                ypos = val - yoff
                font_color = "lightgray"
            else:
                ypos = val + yoff
                font_color = "green"
            
            ax.annotate(xy=(xpos, ypos), s=label,
                       va="center", ha="center",
                       color=font_color, weight="bold",
                       size=14)
            
    return ax

def plot_split_dist(df, column, split_by):
    '''
    Plots the distribution of a feature split by a second feature.
    
    Parameters
    ----------
    df : pandas DataFrame.
    column : str
        The base column for the distribution.
    split_by : str
        The column to split the distribution by.
        
    Returns
    -------
    axes : matplotlib.AxesSubplot.
    
    '''
    fig, ax = plt.subplots()

    for val in df[split_by].unique():
        df_ss = df[df[split_by]==val][column].dropna().copy()
        sns.distplot(df_ss, hist=False, kde=True, label=val, ax=ax)

    ax.set_title(f"Distribution of {column} split by {split_by}")
    ax.legend(title=split_by)
    
    return ax

def plot_counts(df, feature, figsize=(10,5)):
    '''
    Creates an annotated count plot for the given feature.
    
    Parameters
    ----------
    df : pandas DataFrame
    feature : str
    figsize : tuple of ints
        
    Returns
    -------
    axes : matplotlib.AxesSubplot.
    
    '''
    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(df[feature], color="slategray", ax=ax)
    ax.set_title(f"Counts of {feature}")
    values = [p.get_height() for p in ax.patches]
    max_value = max(values)

    for p in ax.patches:
        value_abs = p.get_height()
        value_rel = value_abs / len(df)
        xpos = p.get_x() + p.get_width()/2
        yoff = max_value * 0.05
        if value_abs > 0.8 * max_value:
            ypos = value_abs - yoff
            font_color = "white"
        else:
            ypos = value_abs + yoff
            font_color = "slategray"

        ax.annotate(s=f"{value_abs}\n({100*value_rel:.1f}%)" , 
                    xy=(xpos, ypos),
                    va="center", ha="center",
                    color=font_color, weight="bold",
                    size=10)
    return ax

def predict_and_save_in_kaggle_format(clf, test_df, path_output_folder, name_output_file=None, return_predictions_df=False):
    '''
    Uses the fitted classifier to predict 'Survived' for the given test set and saves the result
    as a Kaggle-compatible .csv file.
    
    Parameters
    ----------
    clf : Classifier object
        The fitted classifier. Has to have a predict method!
    test_df : Pandas DataFrame
        The DataFrame with the test data. Index has to be the PassengerId.
    path_output_folder : String
        The name of the folder the .csv should be save to.
    name_output_file : String, default = None
        If not given, the output file will be named 'predictions_{classifier name}.csv'
    return_predictions_df : Boolean, default = False
        If True, the predictions will be returned as a Pandas DataFrame
        
    Returns
    -------
    Pandas DataFrame : the generated predictions (only returned if return_predictions_df = True)
    
    '''
    if name_output_file==None:
        name_output_file = f"predictions_{clf.__class__.__name__}.csv"
    predictions = pd.DataFrame()
    predictions["PassengerId"] = test_df.index
    predictions["Survived"] = clf.predict(test_df)
    predictions["Survived"] = predictions["Survived"].astype(int)
    predictions.to_csv(os.path.join(path_output_folder, name_output_file), index=False)
    if return_predictions_df:
        return predictions
