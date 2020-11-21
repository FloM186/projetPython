powershell [Reflection.Assembly]::LoadWithPartialName("""System.Windows.Forms""");[Windows.Forms.MessageBox]::show("""Salut, et merci d'utiliser notre App de machine learning Python ! N'oubliez pas de lire le Readme avant utilisation. Cliquez sur "Ok" puis selectionnez le chemin du fichier C:/.../Anaconda/Scripts/activate.bat, et l'App se lancera! Amusez-vous bien!;)""", """App Python Machine learning - M2 SISE""")

rem preparation command
set pwshcmd=powershell -noprofile -command "&{[System.Reflection.Assembly]::LoadWithPartialName('System.windows.forms') | Out-Null;$OpenFileDialog = New-Object System.Windows.Forms.OpenFileDialog; $OpenFileDialog.Title="test";$OpenFileDialog.ShowDialog()|out-null; $OpenFileDialog.FileName}"

rem exec commands powershell and get result in FileName variable
for /f "delims=" %%I in ('%pwshcmd%') do set "FileName=%%I"

echo %FileName%

call %FileName%
bokeh serve --show ProjectApp
PAUSE