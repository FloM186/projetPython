# projetPython
m2 sise python

Il s'agit dans ce projet de realiser une interface graphique permettant d'appliquer des algorithmes de
prediction sur un jeu de donnees. Les utilisateurs de votre outil sont supposes ne rien conna^tre en program-
mation python. Le scenario d'utilisation est le suivant :
1. L'outil prendra en entree un jeu de donnees annote au format csv (avec une virgule pour separateur).
Le jeu de donnees comportera des noms de colonnes a la premiere ligne.
2. L'application permettra a l'utilisateur de denir les variables predictives, et la variable cible.
3. En fonction du type de la variable cible (categorielle ou numerique), l'application proposera au moins
3 algorithmes applicables.
4. L'utilisateur pourra choisir un ou plusieurs de ces algorithmes a appliquer aux donnees. Pour chaque
algorithme a appliquer, les valeurs optimale des hyper-parametres pourront ^etre soit denies par
l'utilisateur, soit identiees automatiquement.
5. Pour chacun des modeles a appliquer, on utilisera une validation croisee et on fournira en sortie les
metriques d'evaluation , le temps de calcul, et surtout une/des gure(s) permettant de percevoir de
facon synthetique la dierence entre les predictions et la realite (Vous pouvez par exemple vous servir
d'une representation factorielle).

Vous serez evalues principalement sur :
| Fonctionnement, 
uidite et facilite de prise en main de l'outil
| Capacite a identier les meilleurs modeles (et parametrages)
| Pertinence et utilite des sorties
| Presentation du rapport et de l'application (interface et code source)
| Temps de calcul


	- Changer data table : convertir en panda df (va falloir voir docu de cette classe table column car lui n'affiche que les table column. Field = nom de colonne. Filer c'est le header, et titre. 
	- Rajouter boxplot
    - Reg linéaire elasticnet
	

Bonus : chemin auto .bat

Mode d'emploi :
Editer le fichier AppPython.bat, et changer le CHEMIN-DU-DOSSIER-ANACONDA avec votre chemin local de votre installation d'anaconda. Pour cela allez sur votre dossier d'installation Anaconda et cliquez droit, "Propriétés" et copier/collez le chemin indiqué. Si par exemple votre Anaconda est installé dans Program Files, la ligne call devrait donner :
> call "C:\Program Files\Anaconda\Scripts\activate.bat"
Une fois cette modification effectuée, enregistrez et fermez le fichier AppPython.bat. Vous pouvez maintenant double cliquer dessus pour lancer automatiquement l'App en html dans votre navigateur internet.

