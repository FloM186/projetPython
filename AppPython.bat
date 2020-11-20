@ECHO OFF
ECHO Si vous avez bien change le chemin de votre installation Anaconda3, une fenetre HTML s'ouvrira dans votre navigateur internet, vous pourrez alors utiliser l'AppPython! Amusez-vous bien!
call "CHEMIN-DU-DOSSIER-ANACONDA\Anaconda\Scripts\activate.bat"
bokeh serve --show ProjectApp
PAUSE