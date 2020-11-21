from os.path import dirname, join
import io

import pandas as pd 

import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve


from statsmodels.api import MNLogit
import statsmodels.api as sm

from bokeh.models.layouts import Column, Row
from bokeh.models.widgets.tables import StringEditor
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ( ColumnDataSource, Title,
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
                    width=600, height=200, sortable=True, 
                    editable=True, fit_columns=True, selectable=True )

# learnig curve pour la regression logistique 
source_learn_curve = ColumnDataSource(data=dict(train_sizes=[], train_mean=[],  
                                                train_mean_p_train_std=[],
                                                train_mean_m_train_std=[],
                                                test_mean=[],
                                                test_mean_p_test_std=[],
                                                test_mean_m_test_std=[]))
learn_curve = figure(title="Accuracy", title_location="left", plot_width=600, plot_height=400)
learn_curve.add_layout(Title(text="number of training examples", align="center"), "below")
learn_curve.varea('train_sizes', 'train_mean_p_train_std','train_mean_m_train_std', source = source_learn_curve, fill_color= "lightskyblue")
learn_curve.line( 'train_sizes', 'train_mean', source = source_learn_curve )
learn_curve.circle( 'train_sizes', 'train_mean', source = source_learn_curve, size = 15, fill_color='blue', legend_label='training accuracy')
learn_curve.varea('train_sizes', 'test_mean_p_test_std','test_mean_m_test_std', source = source_learn_curve, fill_color= "palegreen")
learn_curve.line('train_sizes', 'test_mean', source=source_learn_curve)
learn_curve.circle( 'train_sizes', 'test_mean', source = source_learn_curve, size = 15, fill_color='green', legend_label='validation accuracy')

# Variables de la regression logistique 
controls_reg_log = [var_cible_reg_log_select,var_pred_reg_log_choice, strategy_imputer_reg_log, slider_reg_log_train_test]
for control_reg_log in controls_reg_log:
    control_reg_log.on_change('value', lambda attr,old,new: update_reg_log())

# CallBack des features de la regression logistique 
def update_reg_log():
    # la source de données pour la regression logistique 
    df = pd.read_csv(join(dirname(__file__), 'datasets/'+file_input.filename))

    # la variable cible de la regression logistique
    y = df[var_cible_reg_log_select.value].values

    # les variables cibles de la regression logistique 
    X = df[var_pred_reg_log_choice.value].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=slider_reg_log_train_test.value,
                                                        random_state=1, stratify=y)  

    # encodage des variables cibles categorielles qui ne sont pas numerique
    labelEncoder_y = LabelEncoder()
    if(y.dtype == 'object'):
        y_train = labelEncoder_y.fit_transform(y_train)
        y_test = labelEncoder_y.fit_transform(y_test)
    
    # traitement des valeurs manquantes 
    imputer = SimpleImputer(missing_values=np.nan, strategy = strategy_imputer_reg_log.value)
    imputer = imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    imputer = imputer.fit(X_test)
    X_test = imputer.transform(X_test)

    # regression logistique avec statsmodels
    lr = MNLogit(endog=np.int64(y_train), exog=X_train)
    res = lr.fit()
    
    model = LogisticRegression(random_state=0, multi_class='multinomial', penalty='none', solver='newton-cg').fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    matrix_conf = pd.DataFrame( np.array(confusion_matrix(y_test, y_pred)), index=np.unique(y), columns=[str(i) for i in np.unique(y)] )
    matrix_conf.insert(0, "Observations\Prédictions", np.unique(y), True) 
    
    class_report=classification_report(y_test, y_pred)
    res_lr.text = str(res.summary())+'\n'+str(class_report)+'\n Accuracy score : '+str(accuracy_score(y_test, y_pred))

    # CallBack des colonnes (TableColumn) pour l affichage de la table (DataTable) 
    source_conf.data = {matrix_conf[column_name].name : matrix_conf[column_name] for column_name in get_column_list(matrix_conf)}
    data_conf.source = source_conf
    data_conf.columns = [TableColumn(field = matrix_conf[column_name].name, title = matrix_conf[column_name].name, editor = StringEditor()) for column_name in get_column_list(matrix_conf)]

    # # validation croisee stratifiée
    pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l2', random_state=1,
                                            solver='lbfgs', max_iter=100000))
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
                               X=X_train,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=1)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    source_learn_curve.data = dict( train_sizes=train_sizes, train_mean=train_mean,
                                    train_mean_p_train_std=(train_mean+train_std/3), 
                                    train_mean_m_train_std=(train_mean-train_std/3),
                                    test_mean_p_test_std=(test_mean+test_std/6),
                                    test_mean_m_test_std=(test_mean-test_std/6),
                                    test_mean=test_mean )
    
    
    


# Fin de la regression logistique---------------------------------------------------------------------------------- 








# Regression Linéaire--------------------------------------------------------------------------------------------
# outil pour la selection de la colonne cible pour la régression linéaire
var_cible_reg_lin_select = Select(title="Sélectionner la variable cible :", options = [])
# CallBack du select de la variable prédictive 
def var_cible_reg_lin_select_options(df):
    var_cible_reg_lin_select.options = list(np.append(['------'],get_column_list( df.select_dtypes(include=['object','int64']))))

# Selection des variables predictives 
var_pred_reg_lin_choice = MultiChoice(title="Sélection des variables Prédictives", options=[])

# CallBack des Choix de variables predictives
def var_pred_reg_lin_choice_options(df):
    var_pred_reg_lin_choice.options = list(np.append(['------'],get_column_list( df.select_dtypes(include=['float64','int64']))))

# slider du partitionnement des données test et entrainement 
slider_reg_lin_train_test = Slider(start=0, end=1, value=0.3, step=0.01, title="Proportion des données test" )

# slider du alpha max
slider_reg_lin_alpha = Slider(start=0, end=1, value=0.25, step=0.01, title="Valeur de alpha" )

# slider du alpha pas
slider_reg_lin_alpha_pas = Slider(start=0, end=1, value=0.05, step=0.01, title="Valeur du pas de alpha" )

# select pour les strategies de valeurs manquantes
strategy_imputer_reg_lin = Select(title='Stratégie de remplacement des valeurs manquantes', value='mean', 
                                options=['mean','median','most_frequent'])



# resultat regression linéaire affichage
#res_lr = PreText(text='', width=900, height=700)
res_summ = PreText(text='', width=400)
tableau_alpha =  PreText(text='', width=400)


# Variables de la regression linéaire
controls_reg_lin = [var_cible_reg_lin_select,var_pred_reg_lin_choice, strategy_imputer_reg_lin, slider_reg_lin_train_test]
for control_reg_lin in controls_reg_lin:
    control_reg_lin.on_change('value', lambda attr,old,new: update_reg_lin())


# learnig curve pour la regression logistique 
source_learn_curve = ColumnDataSource(data=dict(train_sizes=[], train_mean=[],  
                                                train_mean_p_train_std=[],
                                                train_mean_m_train_std=[],
                                                test_mean=[],
                                                test_mean_p_test_std=[],
                                                test_mean_m_test_std=[]))
learn_curve = figure(title="Accuracy", title_location="left", plot_width=600, plot_height=400)
learn_curve.add_layout(Title(text="number of training examples", align="center"), "below")
learn_curve.varea('train_sizes', 'train_mean_p_train_std','train_mean_m_train_std', source = source_learn_curve, fill_color= "lightskyblue")
learn_curve.line( 'train_sizes', 'train_mean', source = source_learn_curve )
learn_curve.circle( 'train_sizes', 'train_mean', source = source_learn_curve, size = 15, fill_color='blue', legend_label='training accuracy')
learn_curve.varea('train_sizes', 'test_mean_p_test_std','test_mean_m_test_std', source = source_learn_curve, fill_color= "palegreen")
learn_curve.line('train_sizes', 'test_mean', source=source_learn_curve)
learn_curve.circle( 'train_sizes', 'test_mean', source = source_learn_curve, size = 15, fill_color='green', legend_label='validation accuracy')



# CallBack des features de la regression linéaire
def update_reg_lin():
    # la source de données pour la regression linéaire 
    df = pd.read_csv(join(dirname(__file__), 'datasets/'+file_input.filename))

    # la variable cible de la regression linéaire 
    y = df[var_cible_reg_lin_select.value].values

    # les variables cibles de la regression linéaire 
    X = df[var_pred_reg_lin_choice.value].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=slider_reg_lin_train_test.value,
                                                        random_state=1, stratify=y)  

     # traitement des valeurs manquantes 
    imputer = SimpleImputer(missing_values=np.nan, strategy = strategy_imputer_reg_lin.value)
    imputer = imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    imputer = imputer.fit(X_test)
    X_test = imputer.transform(X_test)
    

    # regression linéaire avec statsmodels
    linreg = sm.OLS(y_train,X_train)
    res = linreg.fit()

    #res_summ.text = str(res.summary())  

    #prediction régression linéaire 
    ypred = linreg.predict(res.params,X_test)

    # figure du nuage de points
    plotprediction = figure(plot_width=900, plot_height=300)
    plotprediction.circle(x=y_test, y=ypred)

    #calcule MSE
    lin_mse = mean_squared_error(y_test,ypred)

   

    #regression linéaire regularisée
    #centrer réduire moyennes d'apprentissage
    sc = StandardScaler()
    X_trainsc = sc.fit_transform(pd.DataFrame(X_train))
    y_trainsc = sc.fit_transform(pd.DataFrame(y_train))
    
    linregu = sm.OLS(y_trainsc,X_trainsc)
    resu = linregu.fit()
    #model = linregu
    #results_fu = resu
    frames = []
    for n in np.arange(0, slider_reg_lin_alpha, slider_reg_lin_alpha_pas).tolist():
        results_fr = linregu.fit_regularized(L1_wt=0, alpha=n, start_params=resu.params)

        results_fr_fit = sm.regression.linear_model.OLSResults(linregu, 
                                                            results_fr.params, 
                                                            linregu.normalized_cov_params)
        frames.append(np.append(results_fr.params, results_fr_fit.ssr))

    df_des_alpha = pd.DataFrame(frames, columns=list(X_train.columns) + ['ssr*'])
    df_des_alpha.index=np.arange(0, slider_reg_lin_alpha, slider_reg_lin_alpha_pas).tolist()
    df_des_alpha.index.name = 'alpha*'
    #affichage
    tableau_alpha = df_des_alpha.T
    tableau_alpha.text = str(tableau_alpha)
    
    #graphiques
    %matplotlib inline

    fig, ax = plt.subplots(1, 2, figsize=(14, 4))

    ax[0] = df6.iloc[:, :-1].plot(ax=ax[0])
    ax[0].set_title('Coefficient')

    ax[1] = df6.iloc[:, -1].plot(ax=ax[1])
    ax[1].set_title('SSR')







# CallBack des features de la regression logistique 
def update_reg_log():
    # la source de données pour la regression logistique 
    df = pd.read_csv(join(dirname(__file__), 'datasets/'+file_input.filename))

    # la variable cible de la regression logistique
    y = df[var_cible_reg_log_select.value].values

    # les variables cibles de la regression logistique 
    X = df[var_pred_reg_log_choice.value].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=slider_reg_log_train_test.value,
                                                        random_state=1, stratify=y)  

    # encodage des variables cibles categorielles qui ne sont pas numerique
    labelEncoder_y = LabelEncoder()
    if(y.dtype == 'object'):
        y_train = labelEncoder_y.fit_transform(y_train)
        y_test = labelEncoder_y.fit_transform(y_test)
    
    # traitement des valeurs manquantes 
    imputer = SimpleImputer(missing_values=np.nan, strategy = strategy_imputer_reg_log.value)
    imputer = imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    imputer = imputer.fit(X_test)
    X_test = imputer.transform(X_test)

    # regression logistique avec statsmodels
    lr = MNLogit(endog=np.int64(y_train), exog=X_train)
    res = lr.fit()
    
    model = LogisticRegression(random_state=0, multi_class='multinomial', penalty='none', solver='newton-cg').fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    matrix_conf = pd.DataFrame( np.array(confusion_matrix(y_test, y_pred)), index=np.unique(y), columns=[str(i) for i in np.unique(y)] )
    matrix_conf.insert(0, "Observations\Prédictions", np.unique(y), True) 
    
    class_report=classification_report(y_test, y_pred)
    res_lr.text = str(res.summary())+'\n'+str(class_report)+'\n Accuracy score : '+str(accuracy_score(y_test, y_pred))

    # CallBack des colonnes (TableColumn) pour l affichage de la table (DataTable) 
    source_conf.data = {matrix_conf[column_name].name : matrix_conf[column_name] for column_name in get_column_list(matrix_conf)}
    data_conf.source = source_conf
    data_conf.columns = [TableColumn(field = matrix_conf[column_name].name, title = matrix_conf[column_name].name, editor = StringEditor()) for column_name in get_column_list(matrix_conf)]

    # # validation croisee stratifiée
    pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l2', random_state=1,
                                            solver='lbfgs', max_iter=100000))
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
                               X=X_train,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=10,
                               n_jobs=1)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    source_learn_curve.data = dict( train_sizes=train_sizes, train_mean=train_mean,
                                    train_mean_p_train_std=(train_mean+train_std/3), 
                                    train_mean_m_train_std=(train_mean-train_std/3),
                                    test_mean_p_test_std=(test_mean+test_std/6),
                                    test_mean_m_test_std=(test_mean-test_std/6),
                                    test_mean=test_mean )


# Fin de la regression linéaire---------------------------------------------------------------------------------- 




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
                                Row(slider_reg_log_train_test, strategy_imputer_reg_log), data_conf, learn_curve, res_lr ), title='Régression Logistique' )
SVM = Panel( child=Row(), title='SVM' )
tabs_methods = Tabs(tabs=[logist, SVM])

layout = column( file_input, tabs_df, data_table, tabs_graphiques, tabs_methods)

curdoc().add_root(layout)
curdoc().title = "Projet Python Cool"
