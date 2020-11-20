from os.path import dirname, join
import io

import pandas as pd 

import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from statsmodels.api import MNLogit

from bokeh.models.layouts import Column, Row
from bokeh.models.widgets.tables import StringEditor
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ( ColumnDataSource, 
                        DataTable, TableColumn, 
                        FileInput, PreText, Select, 
                        Panel, Tabs, MultiChoice, Slider )
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
    df = pd.read_csv(join(dirname(__file__), 'datasets/'+file_input.filename))
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

    # CallBack de la variable cible pour la regression logistique 
    var_cible_reg_log_select_options(df)

    # CallBack des variables prédictives pour la regression logistique 
    var_pred_reg_log_choice_options(df)

# fonction qui retourne les colonnes du dataset
def get_column_list(df):
    column_list=[]
    for i in df.columns:
    	column_list.append(i)
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
    df = pd.read_csv(join(dirname(__file__), 'datasets/'+file_input.filename))
    source_nuage.data = dict( x=df[x_nuage_select.value], y=df[y_nuage_select.value] )
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
    df = pd.read_csv(join(dirname(__file__), 'datasets/'+file_input.filename))

    df_objects = df[ get_column_list(df.select_dtypes(include=['float64','int64'])) ]
    
    unique_elements, count_elements = np.unique(df_objects[hist_quanti_select.value], return_counts=True)
    source_hist_quanti.data = dict(x=unique_elements, top=count_elements)
# Fin histogramme numérique-----------------------------------------------------------------------------------------------


# Regression Logistique--------------------------------------------------------------------------------------------
# outil pour la selection de la colonne cible pour la régression logistique
var_cible_reg_log_select = Select(title="Sélectionner la variable cible :", options = [])
# CallBack du select de la variable prédictive 
def var_cible_reg_log_select_options(df):
    var_cible_reg_log_select.options = list(np.append(['------'],get_column_list( df.select_dtypes(include=['object','int64']))))

# Selection des variables descriptives 
var_pred_reg_log_choice = MultiChoice(title="Sélection des variables Prédictives", options=[])
# CallBack des Choix de variables predictives
def var_pred_reg_log_choice_options(df):
    var_pred_reg_log_choice.options = list(np.append(['------'],get_column_list( df.select_dtypes(include=['float64','int64']))))

# slider du partitionnement des données test et entrainement 
slider_reg_log_train_test = Slider(start=0, end=1, value=0.3, step=0.01, title="Proportion des données test" )

# select pour les strategies de valeurs manquantes
strategy_imputer_reg_log = Select(title='Stratégie de remplacement des valeurs manquantes', value='mean', 
                                options=['mean','median','most_frequent'])

# resultat regression logistique matplotlib
res_lr = PreText(text='', width=900, height=700)

# dataframe pour l affichage de la matrice de confusion
matrix_conf = pd.DataFrame()
source_conf = ColumnDataSource(data=dict(matrix_conf))
columns_conf = []
data_conf = DataTable(source=source_conf, columns = columns,
                    width=600, height=250, sortable=True, 
                    editable=True, fit_columns=True, selectable=True )

# Variables de la regression logistique 
controls_reg_log = [var_cible_reg_log_select,var_pred_reg_log_choice, strategy_imputer_reg_log, slider_reg_log_train_test]
for control_reg_log in controls_reg_log:
    control_reg_log.on_change('value', lambda attr,old,new: update_reg_log())

# CallBack des features de la regression logistique 
def update_reg_log():
    # la source de données pour la regression logistique 
    df = pd.read_csv(join(dirname(__file__), 'datasets/'+file_input.filename))

    print(slider_reg_log_train_test.value)

    # la variable cible de la regression logistique
    y = df[var_cible_reg_log_select.value].values

    # les variables cibles de la regression logistique 
    X = df[var_pred_reg_log_choice.value].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=slider_reg_log_train_test.value,
                                                        random_state=1, stratify=y)  

    labelEncoder_y = LabelEncoder()
    y_train = labelEncoder_y.fit_transform(y_train)
    y_test = labelEncoder_y.fit_transform(y_test)
    
    # traitement des valeurs manquantes 
    imputer = SimpleImputer(missing_values=np.nan, strategy = strategy_imputer_reg_log.value)
    imputer = imputer.fit(X_train)
    X_train = imputer.transform(X_train)

    # regression logitique avec statsmodels
    lr = MNLogit(endog=np.int64(y_train), exog=X_train)
    res = lr.fit()
    
    model = LogisticRegression(random_state=0, multi_class='multinomial', penalty='none', solver='newton-cg').fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    matrix_conf = pd.DataFrame( np.array(confusion_matrix(y_test, y_pred)), index=np.unique(y), columns=np.unique(y) )
    matrix_conf.insert(0, "Observations\Prédictions", np.unique(y), True) 
    
    class_report=classification_report(y_test, y_pred)
    res_lr.text = str(res.summary())+'\n'+str(class_report)+'\n Accuracy score : '+str(accuracy_score(y_test, y_pred))
    print(type( str(accuracy_score(y_test, y_pred)) ))

    # CallBack des colonnes (TableColumn) pour l affichage de la table (DataTable) 
    source_conf.data = {matrix_conf[column_name].name : matrix_conf[column_name] for column_name in get_column_list(matrix_conf)}
    data_conf.source = source_conf
    data_conf.columns = [TableColumn(field = matrix_conf[column_name].name, title = matrix_conf[column_name].name, editor = StringEditor()) for column_name in get_column_list(matrix_conf)]

# Fin de la regression logistique---------------------------------------------------------------------------------- 


# affichage de l'application

# affichage des informations sur le dataset

info_df = Panel(child=Column(df_info), title='Informations sur les variables')
desc_df = Panel(child=Column(df_describe), title='Description du dataset')
tabs_df = Tabs(tabs=[info_df,desc_df])

# affichage des graphiques (nuage de points+histogrammes) pour les variables numeriques
scatter = Panel( child=Column( y_nuage_select, x_nuage_select, nuage ), title='Nuage de points' )
boxplot = Panel( child=Column( hist_quanti_select,hist_quanti ), title='Histogramme des variables numériques' )
tabs_graphiques = Tabs(tabs=[scatter,boxplot])

# affichage des méthodes de machine learning
logist = Panel( child= Column(Row( var_cible_reg_log_select, var_pred_reg_log_choice), 
                                Row(slider_reg_log_train_test, strategy_imputer_reg_log), res_lr, data_conf ), title='Régression Logistique' )
SVM = Panel( child=Row(), title='SVM' )
tabs_methods = Tabs(tabs=[logist, SVM])

layout = column( file_input, tabs_df, data_table, tabs_graphiques, tabs_methods)

curdoc().add_root(layout)
curdoc().title = "Projet Python Cool"



