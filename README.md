**Computation of the acoustic indices to characterize audio WAV files**
>The files pr1.py, pr2.py and pr3.py are modified versions of main_test_indices.py from Guyot and Eldridge repository (https://github.com/patriceguyot/Acoustic_Indices/). These files process, in parallel, the wav files of the Yataros1, Yataros2 and Yataros3 datasets in order to compute their acoustic indices. The Python codes load, from a path, all the wav audios available at http://colecciones.humboldt.org.co/rec/sonidos/publicaciones/MAP/JDT-Yataros/. You should change the content of the variable dir_path according to your local folder name. Notice that the option  Loader=yaml.FullLoader was changed by  Loader=yaml.loader.Loader due to compatibility with python 3.12.2 version. The characterized resulting datasets are saved in csv files whose headers were removed manually for the subsequent processing.
>
**Partition of the characterized datasets according to parts of the day**
 >The SeparationYatarosHalves.py program separates the results from the previous stage into two parts: day and night. The program generates two .csv files for each matrix of acoustic indices that, in turn, correspond to a characterized subset. For example, for Yataros1.csv, the program generates Yataros1Dia.csv and Yataros1Noche.csv and so on.
>
>Similarly, SeparationYatarosQuarters.py separates the matrices of acoustic indices per dataset into four parts: day, afternoon, night and after-midnight. The program generates four .csv files for each matrix of acoustic indices that, in turn, correspond to a characterized subset. For example, for Yataros1.csv, it generates Yataros1Amanecer.csv, Yataros1Anochecer.csv, Yataros1Manana.csv, and Yataros1Tarde.csv and so on.
> 
**Generation of graphical results**
>The Python programs UnsupervisedResultsY1.py, UnsupervisedResultsY2.py and UnsupervisedResultsY3.py, generate plots to visualize the results of the unsupervised feature selection of acoustic indices, where positions in the feature ranking are shown by either colors (heat maps showing the whole feature ranking) or numbers (bar plots showing the results for the top-five ranking).
>
