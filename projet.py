import os
import io
import pandas as pd 

from bokeh.models.layouts import Column
from bokeh.models.widgets.tables import StringEditor
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ( ColumnDataSource, DataTable, TableColumn, FileInput, PreText, Select, Panel, Tabs )
from bokeh.plotting import figure

# initialisation des dataframes pour les ColumnDataSource
df = pd.DataFrame()
source = ColumnDataSource(data=dict(df))
columns = []

# description des jeux de données 
df_info = PreText(text='', width=400)
df_describe = PreText(text='', width=400)

# outil pour la selection de la colonne cible
var_cible_select = Select(title="Sélectionner la variable cible :", options = [])

# initialisation des boutons pour le FileDialog
file_input = FileInput(accept=".csv")

# fonction callback du fileDialog
def update():
    df = pd.read_csv(os.path.abspath(file_input.filename))
    # source = ColumnDataSource(data = dict(df))
    update_df_display(df)

# fonction callback des description des datasets des datasets
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

    # CallBack de la selection de la variable cible
    var_cible_select.options = get_column_list(df)

    # CallBack des parametres du nuage de points 
    set_select(df)

# fonction qui retourne les colonnes du dataset
def get_column_list(df):
		column_list=[]

		for i in df.columns:
			column_list.append(i)
		return column_list

file_input.on_change('filename', lambda attr, old, new: update())

data_table = DataTable( source=source, columns = columns, width=900, height=250, sortable=True, editable=True, fit_columns=True, selectable=True )

# CallBack des options des selects pour le nuage de points
def set_select(df) :
    Y_select.options = get_column_list(df)
    X_select.options = get_column_list(df)

# Nuage de points
Y_select = Select(title="Ordonnées :", options = [])
X_select = Select(title="Abcisses :", options = [])
nuage = figure(plot_width=900, plot_height=300)
nuage.circle()

# Boite à moustaches
bm = figure(plot_width=900, plot_height=300)
bm.line()
# affichage de l'application
scatter = Panel(child=Column(Y_select, X_select, nuage) , title='Nuage de points')
boxplot = Panel(child=bm , title='Boite à moustache')
tabs = Tabs(tabs=[scatter,boxplot])

controls = column(file_input,df_info)
layout = row( controls, column(var_cible_select, data_table, df_describe, tabs ))
curdoc().add_root(layout)
curdoc().title = "Projet Python Cool"