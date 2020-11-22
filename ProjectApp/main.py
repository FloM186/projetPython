from os.path import dirname, join
import io

import pandas as pd 
import numpy as np
import time

from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.linear_model import ElasticNetCV

from statsmodels.api import MNLogit
import statsmodels.api as sm

import itertools  
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Viridis as palette
from bokeh.models.layouts import Column, Row
from bokeh.models.widgets.tables import StringEditor
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ( Spinner, ColumnDataSource, Title,
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
df_info = PreText(text='', width=900)
df_describe = PreText(text='', width=900)

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
    
    # CallBack de la variable cible pour SVM
    var_cible_svm_select_options(df)

    # CallBack des variables prédictives pour SVM 
    var_pred_svm_choice_options(df)

    # CallBack des variables prédictives pour la regression linéaire
    var_pred_reg_lin_choice_options(df)

    # CallBack de la variable cible pour la regression linéaire
    var_cible_reg_lin_select_options(df)
    # CallBack de la variable cible pour KNN
    var_cible_knn_select_options(df)

    # CallBack des variables prédictives pour KNN 
    var_pred_knn_choice_options(df)

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
var_cible_reg_log_select = Select(title="Sélectionner la variable cible ", options = [])

# CallBack du select de la variable prédictive pour les SVMs
def var_cible_reg_log_select_options(df):
    var_cible_reg_log_select.options = list(np.append(['------'],get_column_list( df.select_dtypes(include=['object','int64']))))

# Selection des variables descriptives pour les SVMs
var_pred_reg_log_choice = MultiChoice(title="Sélection des variables Prédictives", options=[])

# CallBack des Choix de variables predictives
def var_pred_reg_log_choice_options(df):
    var_pred_reg_log_choice.options = list(np.append(['------'],get_column_list( df.select_dtypes(include=['float64','int64']))))

# slider du partitionnement des données test et entrainement 
slider_reg_log_train_test = Slider(start=0, end=1, value=0.3, step=0.01, title="Proportion des données test" )

# select pour les strategies de valeurs manquantes
strategy_imputer_reg_log = Select(title='Stratégie de remplacement des valeurs manquantes', value='mean', 
                                options=['mean','median','most_frequent'])

# resultat regression logistique avec statsmodels
res_lr = PreText(text='', width=900, height=700)
temps_lr = PreText(text='', width=900, height=50)
# radioButton pour la pénalité a utiliser pour la regression logistique
penalty = ['l2','l1','elasticnet','none']
penalty_lr_select = Select(title = 'Penalité :', value = 'l2', options=penalty, width=300)

# parametre C du probleme de minisation pour la regression logistique
spinner_c_lr = Spinner(title = 'Paramètre optimisation C', low=0, value=1, step=0.01, width=300)

# nombre de splits dans la validation croisée
spinner_cv_lr = Spinner(title = 'Nombre de split de la validation croisée :', low=1, value=5, step=1, mode='int', width=300)

# dataframe pour l affichage de la matrice de confusion
matrix_conf = pd.DataFrame()
source_conf = ColumnDataSource(data=dict(matrix_conf))
columns_conf = []
data_conf = DataTable(source=source_conf, columns = columns_conf,
                    width=900, height=200, sortable=True, 
                    editable=True, fit_columns=True, selectable=True )

# courbe ROC
source_roc_curve_lr = ColumnDataSource(data=dict(fpr=[], tpr=[]))
roc_curve_lr = figure(title='Courbe ROC', plot_width=900, 
                                            plot_height=400, hidpi=True)
roc_curve_lr.add_layout(Title(text="Taux de Faux Positifs", align="center"), "below")
roc_curve_lr.add_layout(Title(text="Taux de Vrais Positifs", align="center"), "left")
roc_curve_lr.step('fpr', 'tpr', source=source_roc_curve_lr)
roc_curve_lr.circle('fpr', 'tpr', source=source_roc_curve_lr, fill_color='red')

# learnig curve pour la regression logistique 
source_learn_curve_lr = ColumnDataSource(data=dict(train_sizes=[], train_mean=[],  
                                                train_mean_p_train_std=[],
                                                train_mean_m_train_std=[],
                                                test_mean=[],
                                                test_mean_p_test_std=[],
                                                test_mean_m_test_std=[]))
learn_curve_lr = figure(title="Accuracy", title_location="left", plot_width=900, 
                                            plot_height=400, hidpi=True)
learn_curve_lr.add_layout(Title(text="number of training examples", align="center"), "below")
learn_curve_lr.varea('train_sizes', 'train_mean_p_train_std','train_mean_m_train_std',
                                    source = source_learn_curve_lr, fill_color= "lightskyblue")
learn_curve_lr.line( 'train_sizes', 'train_mean', source = source_learn_curve_lr )
learn_curve_lr.circle( 'train_sizes', 'train_mean', source = source_learn_curve_lr,
                                size = 15, fill_color='blue', legend_label='training accuracy')
learn_curve_lr.varea('train_sizes', 'test_mean_p_test_std','test_mean_m_test_std',
                                            source = source_learn_curve_lr, fill_color= "palegreen")
learn_curve_lr.line('train_sizes', 'test_mean', source=source_learn_curve_lr)
learn_curve_lr.circle( 'train_sizes', 'test_mean', source = source_learn_curve_lr,
                                            size = 15, fill_color='green', legend_label='validation accuracy')
learn_curve_lr.legend.location = 'bottom_right'

# Variables de la regression logistique 
controls_reg_log = [var_cible_reg_log_select,
                    var_pred_reg_log_choice, strategy_imputer_reg_log,
                    slider_reg_log_train_test, penalty_lr_select, 
                    spinner_cv_lr, spinner_c_lr]
for control_reg_log in controls_reg_log:
    control_reg_log.on_change('value', lambda attr,old,new: update_reg_log())

# CallBack des features de la regression logistique 
def update_reg_log():
    start_time = time.time()
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
    
    model = LogisticRegression(penalty=penalty_lr_select.value, random_state=1,
                                            C=spinner_c_lr.value, max_iter=10000000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    matrix_conf = pd.DataFrame( np.array(confusion_matrix(y_test, y_pred)),
                                            index=np.unique(y), columns=[str(i) for i in np.unique(y)] )
    matrix_conf.insert(0, "Observations\Prédictions", np.unique(y), True) 
    
    class_report=classification_report(y_test, y_pred)
    res_lr.text = str(res.summary())+'\n'+str(class_report)+'\n Accuracy score : '+str(accuracy_score(y_test, y_pred))

    # CallBack des colonnes (TableColumn) pour l affichage de la table (DataTable) 
    source_conf.data = {matrix_conf[column_name].name : matrix_conf[column_name] for column_name in get_column_list(matrix_conf)}
    data_conf.source = source_conf
    data_conf.columns = [TableColumn(field = matrix_conf[column_name].name, 
                                                    title = matrix_conf[column_name].name, 
                                                    editor = StringEditor()) for column_name in get_column_list(matrix_conf)]

    # # validation croisee stratifiée
    pipe_lr_cv = make_pipeline(StandardScaler(),model)
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr_cv,
                               X=X_train,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=spinner_cv_lr.value,
                               n_jobs=1)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    source_learn_curve_lr.data = dict( train_sizes=train_sizes, train_mean=train_mean,
                                    train_mean_p_train_std=(train_mean+train_std/3), 
                                    train_mean_m_train_std=(train_mean-train_std/3),
                                    test_mean_p_test_std=(test_mean+test_std/6),
                                    test_mean_m_test_std=(test_mean-test_std/6),
                                    test_mean=test_mean )
    
    # Courbe ROC
    probas_lr = model.predict_proba(X_test)
    
    # les taux de vrais et faux positives
    fpr, tpr, _ = roc_curve(labelEncoder_y.fit_transform(y_test), probas_lr[:,0], pos_label=0)
    # get area under the curve
    roc_auc = auc(fpr, tpr) 
    source_roc_curve_lr.data = dict(fpr=fpr, tpr=tpr)
    
    # validation croisée stratifiée
    kfold = StratifiedKFold(n_splits=spinner_cv_lr.value).split(X_train, y_train)
    scores = []
    # rapport lr
    rapport_lr = str(res.summary())+'\n'+str(class_report)+'\n Accuracy score : '+str(accuracy_score(y_test, y_pred))
    rapport_lr = rapport_lr+'\n\n\n\n\n Validation Croisée Stratifiée :'
    
    for k, (train, test) in enumerate(kfold):
        pipe_lr_cv.fit(X_train[train], y_train[train])
        score = pipe_lr_cv.score(X_train[test], y_train[test])
        scores.append(score)
        
        rapport_lr = rapport_lr + '\n          Ensemble: %2d, Class dist.: %s, Accuracy: %.3f' % (k+1,np.bincount(y_train[train]), score)

    # validation croisée
    
    scores = cross_val_score(estimator=pipe_lr_cv,
                                X=X_train,
                                y=y_train,
                                cv=spinner_cv_lr.value,
                                n_jobs=1)
    
    rapport_lr = rapport_lr + '\n Validation Croisée accuracy scores: %s' % list(scores)
    
    rapport_lr = rapport_lr + '\n Validation Croisée accuracy: %.3f +/- %.3f \n\n\n\n\n\n\n\n\n\n' % (np.mean(scores), np.std(scores))
    # rapport de la regression logistique et rapport de la classification
    res_lr.text = rapport_lr

    #temps d'éxcécution
    temps_lr.text = 'Le temps de calcul est de '+str(time.time() - start_time)+' secondes pour cette analyse'
# Fin de la regression logistique---------------------------------------------------------------------------------- 












# Regression Linéaire--------------------------------------------------------------------------------------------
# outil pour la selection de la colonne cible pour la régression linéaire
var_cible_reg_lin_select = Select(title="Sélectionner la variable cible :", options = [])
# CallBack du select de la variable prédictive 
def var_cible_reg_lin_select_options(df):
    var_cible_reg_lin_select.options = list(np.append(['------'],get_column_list( df.select_dtypes(include=['float64','int64']))))

# Selection des variables predictives 
var_pred_reg_lin_choice = MultiChoice(title="Sélection des variables Prédictives", options=[])

# CallBack des Choix de variables predictives
def var_pred_reg_lin_choice_options(df):
    var_pred_reg_lin_choice.options = list(np.append(['------'],get_column_list( df.select_dtypes(include=['float64','int64']))))

# slider du partitionnement des données test et entrainement 
slider_reg_lin_train_test = Slider(start=0, end=1, value=0.3, step=0.01, title="Proportion des données test" )

# slider du alpha max
slider_reg_lin_alpha = Slider(start=0, end=1, value=0.25, step=0.01, title="Valeur de alpha (coefficient de pénalité)" )

# slider du alpha pas
slider_reg_lin_alpha_pas = Slider(start=0, end=1, value=0.05, step=0.01, title="Valeur du pas de alpha" )

# slider de la pénalité L1
slider_reg_lin_L1pen = Slider(start=0, end=1, value=0.45, step=0.01, title="Fraction de la pénalité affectée à L1 : Si 0 = regression ridge, si 1 = regression lasso" )

# select pour les strategies de valeurs manquantes
strategy_imputer_reg_lin = Select(title='Stratégie de remplacement des valeurs manquantes', value='mean', 
                                options=['mean','median','most_frequent'])

# nombre de splits dans la validation croisée
spinner_cv_lin = Spinner(title = 'Nombre de split de la validation croisée :', low=1, value=5, step=1, mode='int', width=300)

# resultat regression linéaire affichage

res_summ = PreText(text='', width=400)
tableau_alpha =  PreText(text='', width=400)
lin_mse = PreText(text='', width=400)
temps_lin = PreText(text='', width=400, height=50)
res_lin = PreText(text='', width=400)
# Variables de la regression linéaire
controls_reg_lin = [var_cible_reg_lin_select,var_pred_reg_lin_choice, strategy_imputer_reg_lin, slider_reg_lin_train_test, slider_reg_lin_alpha, slider_reg_lin_alpha_pas, slider_reg_lin_L1pen]
for control_reg_lin in controls_reg_lin:
    control_reg_lin.on_change('value', lambda attr,old,new: update_reg_lin())



# plot nuage regression
# donnees du nuage de points 
source_nuage_lin = ColumnDataSource(data=dict(x=[], y=[]))
# figure du nuage de points
nuage_lin = figure(plot_width=900, plot_height=300, title="Valeurs observées de l'échantillonnage d'entraînement comparé aux valeurs prédites par le modèle de regression linéaire")
nuage_lin.circle(x='x', y='y',source=source_nuage_lin)
nuage_lin.xaxis.axis_label = 'Valeurs observées'
nuage_lin.yaxis.axis_label = 'Valeurs prédites'

#figure lignes coef alpha
#mapper = linear_cmap(field_name='y', palette=Spectral6 ,low=min(y) ,high=max(y), source=source_lines_reglin)
source_lines_reglin = ColumnDataSource(data=dict(x=[], y=[]))
lines_reglin = figure(plot_width=900, plot_height=300,title="Représentation des coefficients des variables en fonction de la valeur de alpha")
lines_reglin.multi_line('x','y',line_width=2, source=source_lines_reglin)
#colors = itertools.cycle(palette)    
#leg= list(df[var_pred_reg_lin_choice.value].columns)
#for m, color, legende in zip(range(len(df[var_pred_reg_lin_choice.value].columns)), colors, leg):
    #lines_reglin.line(source_lines_reglin.data['x'][m],source_lines_reglin.data['y'][m], legend_label=str(leg).format(m), color=color)
#lines_reglin.legend.location='top_left'
lines_reglin.xaxis.axis_label = 'Valeur de alpha (coefficient de pénalité)'
lines_reglin.yaxis.axis_label = 'Valeur des coefficients'

# CallBack des features de la regression linéaire
def update_reg_lin():
    start_time = time.time()
    # la source de données pour la regression linéaire 
    df = pd.read_csv(join(dirname(__file__), 'datasets/'+file_input.filename))

    # la variable cible de la regression linéaire 
    y = df[var_cible_reg_lin_select.value].values

    # les variables cibles de la regression linéaire 
    X = df[var_pred_reg_lin_choice.value].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=slider_reg_lin_train_test.value, random_state=69)  

    # traitement des valeurs manquantes 
    imputer = SimpleImputer(missing_values=np.nan, strategy = strategy_imputer_reg_lin.value)
    imputer = imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    #X_train = sm.add_constant(X_train)
    imputer = imputer.fit(X_test)
    X_test = imputer.transform(X_test)
    #X_test = sm.add_constant(X_test)

    # regression linéaire avec statsmodels
    linreg = sm.OLS(y_train,X_train)
    res = linreg.fit()
    
    res_summ.text = str(res.summary())  

    

    #prediction régression linéaire 
    y_pred = linreg.predict(res.params,X_test)
    
    #calcule MSE
    lin_mse.text = '\n\n La Mean Squared Error pour ce modèle est de :'+str(mean_squared_error(y_test,y_pred))

    source_nuage_lin.data = dict( x=y_test, y=y_pred)

    frames = []
    for n in np.arange(0, slider_reg_lin_alpha.value, slider_reg_lin_alpha_pas.value).tolist():
        results_fr = linreg.fit_regularized(L1_wt=slider_reg_lin_L1pen.value, alpha=n, start_params=res.params)

        results_fr_fit = sm.regression.linear_model.OLSResults(linreg, 
                                                                results_fr.params, 
                                                                linreg.normalized_cov_params)
        frames.append(np.append(results_fr.params, results_fr_fit.ssr))

        df_des_alpha = pd.DataFrame(frames, columns=list(df[var_pred_reg_lin_choice.value].columns) + ['ssr*'])
    df_des_alpha.index=np.arange(0, slider_reg_lin_alpha.value, slider_reg_lin_alpha_pas.value).tolist()
    df_des_alpha.index.name = 'valeur de alpha :'
    #affichage
    tableau_alphaT = df_des_alpha.T.drop(index="ssr*")
    tableau_alpha.text = str(tableau_alphaT)

    source_lines_reglin.data = dict( 
        x=np.array(
            np.array_split(
                np.array(
                    np.array_split(
                        np.array(list(
                            np.arange(0, slider_reg_lin_alpha.value, slider_reg_lin_alpha_pas.value))*len(tableau_alphaT.index)),
                        len(tableau_alphaT.index)),
                     ).ravel(),
            len(tableau_alphaT.index))
        ).tolist(),

        y=np.array(
            np.array_split(
                np.array(
                    np.array_split(tableau_alphaT.values, len(tableau_alphaT.index))
                    ).ravel(),
            len(tableau_alphaT.index))
        ).tolist())

    print(source_lines_reglin.data)


    #temps d'éxcécution
    temps_lin.text = 'Le temps de calcul est de '+str(time.time() - start_time)+' secondes pour cette analyse'



# Fin de la regression linéaire---------------------------------------------------------------------------------- 



















# Separateur a vastes marges---------------------------------------------------------------------------------------
# outil pour la selection de la colonne cible pour SVM
var_cible_svm_select = Select(title="Sélectionner la variable cible ", options = [])
# CallBack du select de la variable prédictive 
def var_cible_svm_select_options(df):
    var_cible_svm_select.options = list(np.append(['------'],get_column_list( df.select_dtypes(include=['object','int64']))))

# Selection des variables descriptives pour les SVMs
var_pred_svm_choice = MultiChoice(title="Sélection des variables Prédictives", options=[])
# CallBack des Choix de variables predictives
def var_pred_svm_choice_options(df):
    var_pred_svm_choice.options = list(np.append(['------'],get_column_list( df.select_dtypes(include=['float64','int64']))))

# slider du partitionnement des données test et entrainement 
slider_svm_train_test = Slider(start=0, end=1, value=0.3, step=0.01, title="Proportion des données test" )

# select pour les strategies de valeurs manquantes pour SVM
strategy_imputer_svm = Select(title='Stratégie de remplacement des valeurs manquantes', value='mean', 
                                options=['mean','median','most_frequent'])

# resultat regression logistique matplotlib
res_svm = PreText(text='', width=900, height=700)
temps_svm = PreText(text='', width=900, height=50)
# radioButton pour la pénalité a utiliser pour SVM
kernel = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
kernel_svm_select = Select(title = 'Noyau :', value = 'linear', options=kernel, width=300)

# parametre C du probleme de minisation pour SVM
spinner_c_svm = Spinner(title = 'Paramètre optimisation C', low=0, value=1, step=0.01, width=300)

# nombre de splits dans la validation croisée
spinner_cv_svm = Spinner(title = 'Nombre de split de la validation croisée :', low=1, value=5, step=1, mode='int', width=300)

# dataframe pour l affichage de la matrice de confusion
matrix_conf_svm = pd.DataFrame()
source_conf_svm = ColumnDataSource(data=dict(matrix_conf_svm))
columns_conf_svm = []
data_conf_svm = DataTable(source=source_conf_svm, columns = columns_conf_svm,
                    width=900, height=200, sortable=True, 
                    editable=True, fit_columns=True, selectable=True )

# courbe ROC
source_roc_curve_svm = ColumnDataSource(data=dict(fpr_svm=[], tpr_svm=[]))
roc_curve_svm = figure(title='Courbe ROC', plot_width=900, 
                                            plot_height=400, hidpi=True)
roc_curve_svm.add_layout(Title(text="Taux de Faux Positifs", align="center"), "below")
roc_curve_svm.add_layout(Title(text="Taux de Vrais Positifs", align="center"), "left")
roc_curve_svm.step('fpr_svm', 'tpr_svm', source=source_roc_curve_svm)
roc_curve_svm.circle('fpr_svm', 'tpr_svm', source=source_roc_curve_svm, fill_color='red')

# learnig curve pour la regression logistique 
source_learn_curve_svm = ColumnDataSource(data=dict(train_sizes_svm=[], train_mean_svm=[],  
                                                train_mean_p_train_std_svm=[],
                                                train_mean_m_train_std_svm=[],
                                                test_mean_svm=[],
                                                test_mean_p_test_std_svm=[],
                                                test_mean_m_test_std_svm=[]))
learn_curve_svm = figure(title="Accuracy", title_location="left", plot_width=900, 
                                            plot_height=400, hidpi=True)
learn_curve_svm.add_layout(Title(text="number of training examples", align="center"), "below")
learn_curve_svm.varea('train_sizes_svm', 'train_mean_p_train_std_svm','train_mean_m_train_std_svm',
                                    source = source_learn_curve_svm, fill_color= "lightskyblue")
learn_curve_svm.line( 'train_sizes_svm', 'train_mean_svm', source = source_learn_curve_svm )
learn_curve_svm.circle( 'train_sizes_svm', 'train_mean_svm', source = source_learn_curve_svm,
                                size = 15, fill_color='blue', legend_label='training accuracy')
learn_curve_svm.varea('train_sizes_svm', 'test_mean_p_test_std_svm','test_mean_m_test_std_svm',
                                            source = source_learn_curve_svm, fill_color= "palegreen")
learn_curve_svm.line('train_sizes_svm', 'test_mean_svm', source=source_learn_curve_svm)
learn_curve_svm.circle( 'train_sizes_svm', 'test_mean_svm', source = source_learn_curve_svm,
                                            size = 15, fill_color='green', legend_label='validation accuracy')
learn_curve_svm.legend.location = 'bottom_right'

# Variables SVM
controls_svm = [var_cible_svm_select,
                    var_pred_svm_choice, strategy_imputer_svm,
                    slider_svm_train_test, kernel_svm_select, 
                    spinner_cv_svm, spinner_c_svm]
for control_svm in controls_svm:
    control_svm.on_change('value', lambda attr,old,new: update_svm())

# CallBack des features SVMs 
def update_svm():
    start_time = time.time()
    # la source de données pour SVM
    df = pd.read_csv(join(dirname(__file__), 'datasets/'+file_input.filename))

    # la variable cible de SVM
    y = df[var_cible_svm_select.value].values

    # les variables cibles de SVM
    X = df[var_pred_svm_choice.value].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=slider_svm_train_test.value,
                                                        random_state=1, stratify=y)    
    # encodage des variables cibles categorielles qui ne sont pas numerique
    labelEncoder_y = LabelEncoder()
    if(y.dtype == 'object'):
        y_train = labelEncoder_y.fit_transform(y_train)
        y_test = labelEncoder_y.fit_transform(y_test)
    
    # traitement des valeurs manquantes 
    imputer = SimpleImputer(missing_values=np.nan, strategy = strategy_imputer_svm.value)
    imputer = imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    imputer = imputer.fit(X_test)
    X_test = imputer.transform(X_test)

    model = SVC(kernel=kernel_svm_select.value, random_state=1, probability=True,
                                            C=spinner_c_svm.value, max_iter=10000000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    matrix_conf = pd.DataFrame( np.array(confusion_matrix(y_test, y_pred)),
                                            index=np.unique(y), columns=[str(i) for i in np.unique(y)] )
    matrix_conf.insert(0, "Observations\Prédictions", np.unique(y), True) 
    
    class_report=classification_report(y_test, y_pred)

    # CallBack des colonnes (TableColumn) pour l affichage de la table (DataTable) 
    source_conf_svm.data = {matrix_conf[column_name].name : matrix_conf[column_name] for column_name in get_column_list(matrix_conf)}
    data_conf_svm.source = source_conf_svm
    data_conf_svm.columns = [TableColumn(field = matrix_conf[column_name].name, 
                                                    title = matrix_conf[column_name].name, 
                                                    editor = StringEditor()) for column_name in get_column_list(matrix_conf)]
    
    # # validation croisee stratifiée
    pipe_svm_cv = make_pipeline(StandardScaler(),model)
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_svm_cv,
                               X=X_train,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1.0, 10),
                               cv=spinner_cv_svm.value,
                               n_jobs=1)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    source_learn_curve_svm.data = dict( train_sizes_svm=train_sizes, train_mean_svm=train_mean,
                                    train_mean_p_train_std_svm=(train_mean+train_std/3), 
                                    train_mean_m_train_std_svm=(train_mean-train_std/3),
                                    test_mean_p_test_std_svm=(test_mean+test_std/6),
                                    test_mean_m_test_std_svm=(test_mean-test_std/6),
                                    test_mean_svm=test_mean )
    
    # Courbe ROC
    probas_svm = model.predict_proba(X_test)
    
    # les taux de vrais et faux positives
    fpr, tpr, _ = roc_curve(labelEncoder_y.fit_transform(y_test), probas_svm[:,0], pos_label=0)
    # get area under the curve
    roc_auc = auc(fpr, tpr) 
    source_roc_curve_svm.data = dict(fpr_svm=fpr, tpr_svm=tpr)
    
    # validation croisée stratifiée
    kfold = StratifiedKFold(n_splits=spinner_cv_svm.value).split(X_train, y_train)
    scores = []
    # rapport svm
    rapport_svm = 'Les indices des Vecteurs Supports : '+str(model.support_)+'\n'+'\n Vecteurs Supports : '+str(list(model.support_vectors_))
    rapport_svm = rapport_svm+'\n'+str(class_report)+'\n Accuracy score : '+str(accuracy_score(y_test, y_pred))
    rapport_svm = rapport_svm+'\n\n\n\n\n Validation Croisée Stratifiée :'

    for k, (train, test) in enumerate(kfold):
        pipe_svm_cv.fit(X_train[train], y_train[train])
        score = pipe_svm_cv.score(X_train[test], y_train[test])
        scores.append(score)
        
        rapport_svm = rapport_svm + '\n          Ensemble: %2d, Class dist.: %s, Accuracy: %.3f' % (k+1,np.bincount(y_train[train]), score)
    
    # validation croisée
    
    scores = cross_val_score(estimator=pipe_svm_cv,
                                X=X_train,
                                y=y_train,
                                cv=spinner_cv_svm.value,
                                n_jobs=1)
    
    rapport_svm = rapport_svm + '\n Validation Croisée accuracy scores: %s' % list(scores)
    
    rapport_svm = rapport_svm + '\n Validation Croisée accuracy: %.3f +/- %.3f \n\n\n\n\n\n\n\n\n\n' % (np.mean(scores), np.std(scores))
    
    # rapport de la regression logistique et rapport de la classification
    res_svm.text = rapport_svm
    #temps d'éxcécution
    temps_svm.text = 'Le temps de calcul est de '+str(time.time() - start_time)+' secondes pour cette analyse'
# Fin de Separateur a vastes marges  ------------------------------------------------------------------------------







# K plus proches voisins-------------------------------------------------------------------------------------------
# outil pour la selection de la colonne cible pour KNN
var_cible_knn_select = Select(title="Sélectionner la variable cible ", options = [])
# CallBack du select de la variable prédictive 
def var_cible_knn_select_options(df):
    var_cible_knn_select.options = list(np.append(['------'],get_column_list( df.select_dtypes(include=['object','int64']))))

# Selection des variables descriptives pour les SVMs
var_pred_knn_choice = MultiChoice(title="Sélection des variables Prédictives", options=[])
# CallBack des Choix de variables predictives
def var_pred_knn_choice_options(df):
    var_pred_knn_choice.options = list(np.append(['------'],get_column_list( df.select_dtypes(include=['float64','int64']))))

# slider du partitionnement des données test et entrainement 
slider_knn_train_test = Slider(start=0, end=1, value=0.3, step=0.01, title="Proportion des données test" )

# select pour les strategies de valeurs manquantes pour SVM
strategy_imputer_knn = Select(title='Stratégie de remplacement des valeurs manquantes', value='mean', 
                                options=['mean','median','most_frequent'])

# resultat regression logistique matplotlib
res_knn = PreText(text='', width=900, height=700)
temps_knn = PreText(text='', width=900, height=50)
# parametre n du nombre de voisins 
spinner_n_knn = Spinner(title = 'Nombre de voisins', low=0, value=5, step=1, mode='int', width=300)

# nombre de splits dans la validation croisée
spinner_cv_knn = Spinner(title = 'Nombre de split de la validation croisée :', low=1, value=5, step=1, mode='int', width=300)

# dataframe pour l affichage de la matrice de confusion
matrix_conf_knn = pd.DataFrame()
source_conf_knn = ColumnDataSource(data=dict(matrix_conf_knn))
columns_conf_knn = []
data_conf_knn = DataTable(source=source_conf_knn, columns = columns_conf_knn,
                    width=900, height=200, sortable=True, 
                    editable=True, fit_columns=True, selectable=True )

# courbe ROC
source_roc_curve_knn = ColumnDataSource(data=dict(fpr_knn=[], tpr_knn=[]))
roc_curve_knn = figure(title='Courbe ROC', plot_width=900, 
                                            plot_height=400, hidpi=True)
roc_curve_knn.add_layout(Title(text="Taux de Faux Positifs", align="center"), "below")
roc_curve_knn.add_layout(Title(text="Taux de Vrais Positifs", align="center"), "left")
roc_curve_knn.step('fpr_knn', 'tpr_knn', source=source_roc_curve_knn)
roc_curve_knn.circle('fpr_knn', 'tpr_knn', source=source_roc_curve_knn, fill_color='red')

# learnig curve pour la regression logistique 
source_learn_curve_knn = ColumnDataSource(data=dict(train_sizes_knn=[], train_mean_knn=[],  
                                                train_mean_p_train_std_knn=[],
                                                train_mean_m_train_std_knn=[],
                                                test_mean_knn=[],
                                                test_mean_p_test_std_knn=[],
                                                test_mean_m_test_std_knn=[]))
learn_curve_knn = figure(title="Accuracy", title_location="left", plot_width=900, 
                                            plot_height=400, hidpi=True)
learn_curve_knn.add_layout(Title(text="number of training examples", align="center"), "below")
learn_curve_knn.varea('train_sizes_knn', 'train_mean_p_train_std_knn','train_mean_m_train_std_knn',
                                    source = source_learn_curve_knn, fill_color= "lightskyblue")
learn_curve_knn.line( 'train_sizes_knn', 'train_mean_knn', source = source_learn_curve_knn )
learn_curve_knn.circle( 'train_sizes_knn', 'train_mean_knn', source = source_learn_curve_knn,
                                size = 15, fill_color='blue', legend_label='training accuracy')
learn_curve_knn.varea('train_sizes_knn', 'test_mean_p_test_std_knn','test_mean_m_test_std_knn',
                                            source = source_learn_curve_knn, fill_color= "palegreen")
learn_curve_knn.line('train_sizes_knn', 'test_mean_knn', source=source_learn_curve_knn)
learn_curve_knn.circle( 'train_sizes_knn', 'test_mean_knn', source = source_learn_curve_knn,
                                            size = 15, fill_color='green', legend_label='validation accuracy')
learn_curve_knn.legend.location = 'bottom_right'

# Variables KNN
controls_knn = [var_cible_knn_select,
                    var_pred_knn_choice, strategy_imputer_knn,
                    slider_knn_train_test,
                    spinner_cv_knn, spinner_n_knn]
for control_knn in controls_knn:
    control_knn.on_change('value', lambda attr,old,new: update_knn())

# CallBack des features SVMs 
def update_knn():
    start_time = time.time()
    # la source de données pour SVM
    df = pd.read_csv(join(dirname(__file__), 'datasets/'+file_input.filename))


    # la variable cible de SVM
    y = df[var_cible_knn_select.value].values

    # les variables cibles de SVM
    X = df[var_pred_knn_choice.value].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=slider_knn_train_test.value,
                                                        random_state=1, stratify=y)
    
    # encodage des variables cibles categorielles qui ne sont pas numerique
    labelEncoder_y = LabelEncoder()
    if(y.dtype == 'object'):
        y_train = labelEncoder_y.fit_transform(y_train)
        y_test = labelEncoder_y.fit_transform(y_test)
    
    # traitement des valeurs manquantes 
    imputer = SimpleImputer(missing_values=np.nan, strategy = strategy_imputer_knn.value)
    imputer = imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    imputer = imputer.fit(X_test)
    X_test = imputer.transform(X_test)

    model = KNeighborsClassifier(n_neighbors=spinner_n_knn.value)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    matrix_conf = pd.DataFrame( np.array(confusion_matrix(y_test, y_pred)),
                                            index=np.unique(y), columns=[str(i) for i in np.unique(y)] )
    matrix_conf.insert(0, "Observations\Prédictions", np.unique(y), True)

    class_report=classification_report(y_test, y_pred)

    # CallBack des colonnes (TableColumn) pour l affichage de la table (DataTable) 
    source_conf_knn.data = {matrix_conf[column_name].name : matrix_conf[column_name] for column_name in get_column_list(matrix_conf)}
    data_conf_knn.source = source_conf_knn
    data_conf_knn.columns = [TableColumn(field = matrix_conf[column_name].name, 
                                                    title = matrix_conf[column_name].name, 
                                                    editor = StringEditor()) for column_name in get_column_list(matrix_conf)]
    
    # # validation croisee stratifiée
    pipe_knn_cv = make_pipeline(StandardScaler(),model)
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_knn_cv,
                                X=X_train, y=y_train, 
                                train_sizes=np.linspace(0.1, 1.0, 10),
                                cv=spinner_cv_knn.value,
                                n_jobs=1)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    source_learn_curve_knn.data = dict( train_sizes_knn=train_sizes, train_mean_knn=train_mean,
                                    train_mean_p_train_std_knn=(train_mean+train_std/3), 
                                    train_mean_m_train_std_knn=(train_mean-train_std/3),
                                    test_mean_p_test_std_knn=(test_mean+test_std/6),
                                    test_mean_m_test_std_knn=(test_mean-test_std/6),
                                    test_mean_knn=test_mean )
    
     
    # Courbe ROC
    probas_knn = model.predict_proba(X_test)

    # les taux de vrais et faux positives
    fpr, tpr, _ = roc_curve(labelEncoder_y.fit_transform(y_test), probas_knn[:,0], pos_label=0)
    # get area under the curve
    roc_auc = auc(fpr, tpr) 
    source_roc_curve_knn.data = dict(fpr_knn=fpr, tpr_knn=tpr)

    # validation croisée stratifiée
    kfold = StratifiedKFold(n_splits=spinner_cv_knn.value).split(X_train, y_train)
    scores = []
    # rapport knn
    rapport_knn = "Les Classes détectés par l'algorithme:"+str(model.classes_)+'\n'+'La métrique utilisée : '+str(model.effective_metric_)+'\n'+str(class_report)+'\n Accuracy score : '+str(accuracy_score(y_test, y_pred))
    rapport_knn = rapport_knn+'\n\n\n\n\n Validation Croisée Stratifiée :'

    for k, (train, test) in enumerate(kfold):
        pipe_knn_cv.fit(X_train[train], y_train[train])
        score = pipe_knn_cv.score(X_train[test], y_train[test])
        scores.append(score)
        
        rapport_knn = rapport_knn + '\n          Ensemble: %2d, Class dist.: %s, Accuracy: %.3f' % (k+1,np.bincount(y_train[train]), score)
    
    scores = cross_val_score(estimator=pipe_knn_cv,
                                X=X_train,
                                y=y_train,
                                cv=spinner_cv_knn.value,
                                n_jobs=1)
    
    rapport_knn = rapport_knn + '\n Validation Croisée accuracy scores: %s' % list(scores)
    
    rapport_knn = rapport_knn + '\n Validation Croisée accuracy: %.3f +/- %.3f \n\n\n\n\n\n\n\n\n\n' % (np.mean(scores), np.std(scores))
    
    # rapport de la regression logistique et rapport de la classification
    res_knn.text = rapport_knn
     #temps d'éxcécution
    temps_knn.text = 'Le temps de calcul est de '+str(time.time() - start_time)+' secondes pour cette analyse'
# fin K plus proches voisins-------------------------------------------------------------------------------------------




# affichage de l'application

# affichage des informations sur le dataset

info_df = Panel(child=Column(df_info), title='Informations sur les variables')
desc_df = Panel(child=Column(df_describe), title='Description du dataset')
tabs_df = Tabs(tabs=[info_df,desc_df])

# affichage des graphiques (nuage de points+histogrammes) pour les variables numeriques
scatter = Panel( child=Column( y_nuage_select, x_nuage_select, nuage ), title='Nuage de points' )
boxplot = Panel( child=Column( hist_quanti_select,hist_quanti ), title='Histogramme des variables numériques' )
tabs_graphiques = Tabs(tabs=[scatter,boxplot], width=900)

# affichage de la regression logistique
logist = Panel( child= Column(Row( var_cible_reg_log_select, var_pred_reg_log_choice), 
                                Row(slider_reg_log_train_test, strategy_imputer_reg_log),
                                Row(penalty_lr_select, spinner_c_lr, spinner_cv_lr),
                                 temps_lr,data_conf,roc_curve_lr,learn_curve_lr, res_lr), title='Régression Logistique' )


#affichage regression linéaire
reglineaire = Panel( child= Column(Row(var_cible_reg_lin_select, var_pred_reg_lin_choice), 
                                Row(slider_reg_lin_train_test, strategy_imputer_reg_lin),
                                Row(slider_reg_lin_alpha, slider_reg_lin_alpha_pas, slider_reg_lin_L1pen),
                                temps_lin, res_summ, lin_mse, nuage_lin, tableau_alpha, lines_reglin, res_lin), title='Régression Linéaire Lasso/Ridge/ElasticNet' )

# affichage des SVM
SVM= Panel( child= Column(Row( var_cible_svm_select, var_pred_svm_choice), 
                                Row(slider_svm_train_test, strategy_imputer_svm),
                                Row(kernel_svm_select, spinner_c_svm, spinner_cv_svm),
                                temps_svm,data_conf_svm,roc_curve_svm,learn_curve_svm, res_svm), title='Séparateurs à vastes marges' )


# affichage de la KNN
KNN= Panel( child= Column(Row( var_cible_knn_select, var_pred_knn_choice), 
                                Row(slider_knn_train_test, strategy_imputer_knn),
                                Row(spinner_n_knn, spinner_cv_knn),
                                temps_knn,data_conf_knn,roc_curve_knn,learn_curve_knn, res_knn), title='K Plus Proches Voisins' )

tabs_methods = Tabs(tabs=[logist, SVM, KNN, reglineaire], width=900)

layout = column( file_input, tabs_df, data_table, tabs_graphiques, tabs_methods)

curdoc().add_root(layout)
curdoc().title = "Projet Python Cool"
