**Generate the indices calculation results**
>The files pr1.py, pr2.py and pr3.py are modified versions of main_test_indices.py from Guyot and Eldridge repository (https://github.com/patriceguyot/Acoustic_Indices/). These files process, in parallel, the wav files of the Yataros1, Yataros2 and Yataros3 datasets in order to compute their acoustic indices. The Python codes load, from a path, all the wav audios available at http://colecciones.humboldt.org.co/rec/sonidos/publicaciones/MAP/JDT-Yataros/. You should change the content of the variable dir_path according to your local folder name. Notice that the option  Loader=yaml.FullLoader was changed by  Loader=yaml.loader.Loader due to compatibility with python 3.12.2 version. The characterized resulting datasets are saved in csv files whose headers were removed manually for the subsequent processing.
>
**Partitioning of indices results**
 > SeparationYatarosHalves.py program separte the indices results in two parts: day and night.  The program generate two .csv files for each dataset indices result. For example, for Yataros1.csv, generate Yataros1Dia.csv and Yataros1Noche.csv and so on.
 > SeparationYatarosQuarters.py separate the indices results in four parts: day, afterrnoon, night and AfterMidnight. The program generate four .csv files for each dataset indices result. For example, for Yataros1.csv, generate Yataros1Amanecer.csv,      Yataros1Anochecer.csv, Yataros1Manana.csv,  and Yataros1Tarde.csv and so on.
> 
**Generating graphics results**
>The Python programs UnsupervisedResultsY1.py, UnsupervisedResultsY2.py and UnsupervisedResultsY3.py, generate the plots obtained from the index results for each partition. The plots are shown in the paper entitled "       ". Each Python program generates 8 pdf plots for Yataros1, Yataros2 and Yataros3. Two calorie chart showing the ranking of the 40 indices and six bar plots showing the top five ranked indices.
