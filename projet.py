import os
import io

import pandas as pd 

import numpy as np

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from bokeh.models.layouts import Column, Row
from bokeh.models.widgets.tables import StringEditor
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ( ColumnDataSource, 
                        DataTable, TableColumn, 
                        FileInput, PreText, Select, 
                        Panel, Tabs, MultiChoice )
from bokeh.plotting import figure


# DataSet------------------------------------------------------------------------
# initialisation des dataframes pour les ColumnDataSources
df = pd.DataFrame()
source = ColumnDataSource(data=dict(df))
columns = []

# description des jeux de données 
df_info = PreText(text='', width=400)
df_describe = PreText(text='', width=400)

# initialisation des boutons pour le FileDialog
file_input = FileInput(accept=".csv")

# fonction callback du FileDialog (file_input)
def update():
    df = pd.read_csv(os.path.abspath(file_input.filename))
    # source = ColumnDataSource(data = dict(df))
    update_df_display(df)

# fonction callback des description des datasets
def update_df_display(df):
    # CallBack infos sur le dataset
    buf = io.StringIO()
    df.info(buf = buf)
    s = buf.getvalue()
    df_info.text = str(s)

    # CallBack description du datasets
    df_describe.text = str(df.describe())
    
    # CallBack des colonnes (TableColumn) pour l affichage de la table (DataTable) 
    source.data = {df[column_name].name : df[column_name] for column_name in get_column_list(df)}
    data_table.source = source
    data_table.columns = [TableColumn(field = df[column_name].name, title = df[column_name].name, editor = StringEditor()) for column_name in get_column_list(df)]

    # CallBack des Selects du nuage de points (mise a jour des variables)
    nuage_var_select(df)

    # CallBack des Selects du nuage de points (mise a jour des variables)
    hist_quali_var_select(df)

    # CallBack des variables prédictives
    var_pred_choice_options(df)

# fonction qui retourne les colonnes du dataset
def get_column_list(df):
    column_list=[]
    for i in df.columns:
    	column_list.append(i)
    # column_list.insert(0,'----------')
    return column_list


file_input.on_change('filename', lambda attr, old, new: update())

data_table = DataTable( source=source, columns = columns,
                         width=900, height=250, sortable=True, 
                         editable=True, fit_columns=True, selectable=True )
# Fin DataSet----------------------------------------------------------------------     



# Nuage de points------------------------------------------------------------------
# selection des axes abscisse et ordonnées
y_nuage_select = Select(title="Ordonnées :", options = [])
x_nuage_select = Select(title="Abcisses :", options = [])

# donnees du nuage de points 
source_nuage = ColumnDataSource(data=dict(x=[], y=[]))

# figure du nuage de points
nuage = figure(plot_width=900, plot_height=300)
nuage.circle(x='x', y='y', source=source_nuage)

# controle du nuage de points (appels CallBacks)
controls_nuage = [x_nuage_select,y_nuage_select]
for control_nuage in controls_nuage : 
    control_nuage.on_change('value',lambda attr,old,new : update_nuage())

# CallBack des options des selects pour le nuage de points
def nuage_var_select(df) :
    y_nuage_select.options = list(np.append(['------'],get_column_list( df.select_dtypes(include=['float64','int64']))))
    x_nuage_select.options = list(np.append(['------'],get_column_list( df.select_dtypes(include=['float64','int64']))))
# CallBack du nuage de points 
def update_nuage():
    df = pd.read_csv(os.path.abspath(file_input.filename))
    source_nuage.data = dict(x=df[x_nuage_select.value],y=df[y_nuage_select.value])
# Fin nuage de points----------------------------------------------------------------------------------


# Histogramme numérique------------------------------------------------------------------------
# selection de la variable numérique pour l histogramme 
hist_quanti_select = Select(title="Choisir une variable :", options = [])

# donnees pour histogramme 
source_hist_quanti= ColumnDataSource(data=dict(x=[], top=[]))

# figure de l histogramme
hist_quanti = figure(plot_width=900, plot_height=300)
hist_quanti.vbar(x='x', top='top', source=source_hist_quanti, width=0.5)

hist_quanti_select.on_change('value',lambda attr,old,new : update_hist_quali())

# CallBack des options des selects pour l histogramme numérique 
def hist_quali_var_select(df) :
    hist_quanti_select.options = list(np.append(['------'],get_column_list( df.select_dtypes(include=['float64','int64']))))

# CallBack du nuage de points 
def update_hist_quali():
    df = pd.read_csv(os.path.abspath(file_input.filename))

    df_objects = df[ get_column_list(df.select_dtypes(include=['float64','int64'])) ]
    
    unique_elements, count_elements = np.unique(df_objects[hist_quanti_select.value], return_counts=True)
    source_hist_quanti.data = dict(x=unique_elements, top=count_elements)
# Fin histogramme numérique-----------------------------------------------------------------------------------------------


# Regression Logistique--------------------------------------------------------------------------------

# Selection des variables descriptives 
var_pred_choice = MultiChoice(title="Selection des variables Prédictives", options=[])

# CallBack des Choix de variables predictives
def var_pred_choice_options(df):
    var_pred_choice.options = get_column_list(df)

# outil pour la selection de la colonne cible pour la régression logistique
var_cible_select = Select(title="Sélectionner la variable cible :", options = [])

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    
    # setup marker et color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot de la surface de decision
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test examples
    if test_idx:
        # plot tout les exemples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')


# affichage de l'application

# affichage des graphiques (nuage de points+histogrammes) pour les variables numeriques
scatter = Panel( child=Column( y_nuage_select, x_nuage_select, nuage ), title='Nuage de points' )
boxplot = Panel( child=Column( hist_quanti_select,hist_quanti ), title='Histogramme des variables numériques' )
tabs_graphiques = Tabs(tabs=[scatter,boxplot])

# affichage des méthodes de machine learning
logist = Panel( child=Row( var_cible_select, var_pred_choice ), title='Régression Logistique' )
SVM = Panel( child=Row(), title='SVM' )
tabs_methods = Tabs(tabs=[logist, SVM])

table = Panel(child=Row(data_table), title='Tableau de données')
info = Panel(child=Row(df_info), title='Informations sur les données')
desc = Panel(child=Row(df_describe), title='Description des données')
tabs_description = Tabs(tabs=[table, info, desc])
controls = column(file_input)
layout = row( controls, column(tabs_description, tabs_graphiques, tabs_methods))

curdoc().add_root(layout)
curdoc().title = "Projet Python Cool"