# importing libraries
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd
from IPython.display import display
from pandas.plotting import scatter_matrix
import seaborn as sns

filename = "D:/UNALAcademico/2023/Proyecto/proyectoPTA/unsupervised/Results/ClasificacionIndicesYataros3.csv"
filename1 = "D:/UNALAcademico/2023/Proyecto/proyectoPTA/unsupervised/Results/ClasificacionIndicesYataros3Labels3T.csv"



#Pandas
indexesRes1 = pd.read_csv(filename1)
display(indexesRes1)

#NP array
indexesRes = np.loadtxt(filename, delimiter=',', dtype=int, encoding="utf-8-sig")


indices=np.ones((len(indexesRes), len(indexesRes[0]))) * -1
for i in range(len(indexesRes)):
    for j in range(len(indexesRes[0])):
        indices[i,j]=list(indexesRes[i,:]).index(j)


#indexesRes = np.loadtxt(filename, delimiter=',')
#print(indexesRes)
print('Tamaño= (',len(indexesRes), len(indexesRes[0]),')')
input()


#indexesResMetodos=np.array(indexesResMetodos)

filas = len(indexesRes)
columnas = len(indexesRes[0])
print("x", columnas)
print("y", filas)
indexesResMetodos=np.zeros((filas,columnas))

def procesarPorMetodos():
    j=0
    cont=0
    while j < 6:
        #i=j
        for i in range(j,len(indices),6):
            print ("i",i)
            indexesResMetodos[cont]=indices[i]
            cont=cont+1
        j=j+1
        print("j",j)
    print(indexesResMetodos)
   

def CorrelationYataros3Bar():
    #labels = list(range(35,40))
    labels = ["5th Best", "4th Best", "3rd Best", "2nd Best",  "Best"]
    VAR = [26, 8, 21, 12, 7]
    MAD = [26, 3, 17, 14, 9]
    DR = [2,1,38,13,24]
    LS = [16,3,8,9,24]
    PCA =[4,24,18,12,29]
    NMF = [25,10,27,16,6]
    
    #x = range(35,40)
    VAR = [ int (i) for i in indices[0, 35:40] ]   #int(list(indices[0, 35:40])) #1
    MAD = [ int (i) for i in indices[1, 35:40] ] #2
    DR = [ int (i) for i in indices[2, 35:40] ] #3
    LS = [ int (i) for i in indices[3, 35:40] ] #4
    PCA = [ int (i) for i in indices[4, 35:40] ] #5
    NMF = [ int (i) for i in indices[5, 35:40] ] #6

    x = np.arange(len(labels))  # the label locations
    width = 0.14  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - (3*width-width/2), VAR, width, label='VAR')
    rects2 = ax.bar(x - (2*width-width/2), MAD, width, label='MAD')
    rects3 = ax.bar(x - width/2, DR, width, label='DR')
    rects4 = ax.bar(x + width/2, LS, width, label='LS')
    rects5 = ax.bar(x + (2*width-width/2), PCA, width, label='PCA')
    rects6 = ax.bar(x + (3*width-width/2), NMF, width, label='NMF')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Acoustic feature identifier',fontsize='14')
    ax.set_xlabel('Ranking position of the top-five best features',fontsize='14')
    #ax.set_title('YAT3',fontsize='14')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width()/6, height),
                        xytext=(4, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', size=12, rotation=90)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    autolabel(rects6)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=6, fontsize=12)
    fig.tight_layout()
    plt.savefig('D://UNALAcademico//2023//Proyecto//proyectoPTA//unsupervised//Yataros3_barChart_B5.pdf') 
    plt.show()
    #plt.savefig("D://UNALAcademico//2023//Proyecto//proyectoPTA//unsupervised//Yataros3B5Bar.pdf", format='pdf', bbox_inches= 'tight')
    
    
#----------------------------------------------------------------------------------------------------------------------#    
def CorrelationYataros3DiaBar():
    #labels = list(range(35,40))
    labels = ["5th Best", "4th Best", "3rd Best", "2nd Best",  "Best"]
    
    #x = range(35,40)
    VAR = [ int (i) for i in indices[6, 35:40] ]   #int(list(indices[0, 35:40])) #1
    MAD = [ int (i) for i in indices[7, 35:40] ] #2
    DR = [ int (i) for i in indices[8, 35:40] ] #3
    LS = [ int (i) for i in indices[9, 35:40] ] #4
    PCA = [ int (i) for i in indices[10, 35:40] ] #5
    NMF = [ int (i) for i in indices[11, 35:40] ] #6

    x = np.arange(len(labels))  # the label locations
    width = 0.14  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - (3*width-width/2), VAR, width, label='VAR')
    rects2 = ax.bar(x - (2*width-width/2), MAD, width, label='MAD')
    rects3 = ax.bar(x - width/2, DR, width, label='DR')
    rects4 = ax.bar(x + width/2, LS, width, label='LS')
    rects5 = ax.bar(x + (2*width-width/2), PCA, width, label='PCA')
    rects6 = ax.bar(x + (3*width-width/2), NMF, width, label='NMF')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Acoustic feature identifier',fontsize='14')
    ax.set_xlabel('Ranking position of the top-five best features',fontsize='14')
    #ax.set_title('YAT3',fontsize='14')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width()/6, height),
                        xytext=(4, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', size=12, rotation=90)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    autolabel(rects6)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=6, fontsize=12)
    fig.tight_layout()
    plt.savefig('D://UNALAcademico//2023//Proyecto//proyectoPTA//unsupervised//Yataros3Dia_barChart_B5.pdf') 
    plt.show()

#----------------------------------------------------------------------------------------------------------------------#            
def CorrelationYataros3NocheBar():
    
    labels = ["5th Best", "4th Best", "3rd Best", "2nd Best",  "Best"]
    
    #x = range(35,40)
    VAR = [ int (i) for i in indices[12, 35:40] ]   #int(list(indices[0, 35:40])) #1
    MAD = [ int (i) for i in indices[13, 35:40] ] #2
    DR = [ int (i) for i in indices[14, 35:40] ] #3
    LS = [ int (i) for i in indices[15, 35:40] ] #4
    PCA = [ int (i) for i in indices[16, 35:40] ] #5
    NMF = [ int (i) for i in indices[17, 35:40] ] #6

    x = np.arange(len(labels))  # the label locations
    width = 0.14  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - (3*width-width/2), VAR, width, label='VAR')
    rects2 = ax.bar(x - (2*width-width/2), MAD, width, label='MAD')
    rects3 = ax.bar(x - width/2, DR, width, label='DR')
    rects4 = ax.bar(x + width/2, LS, width, label='LS')
    rects5 = ax.bar(x + (2*width-width/2), PCA, width, label='PCA')
    rects6 = ax.bar(x + (3*width-width/2), NMF, width, label='NMF')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Acoustic feature identifier',fontsize='14')
    ax.set_xlabel('Ranking position of the top-five best features',fontsize='14')
    #ax.set_title('YAT3',fontsize='14')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width()/6, height),
                        xytext=(4, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', size=12, rotation=90)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    autolabel(rects6)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=6, fontsize=12)
    fig.tight_layout()
    plt.savefig('D://UNALAcademico//2023//Proyecto//proyectoPTA//unsupervised//Yataros3Noche_barChart_B5.pdf') 
    plt.show()


#----------------------------------------------------------------------------------------------------------------------#
def CorrelationYataros3MorningBar():
    
    labels = ["5th Best", "4th Best", "3rd Best", "2nd Best",  "Best"]
    
    #x = range(35,40)
    VAR = [ int (i) for i in indices[18, 35:40] ]   #int(list(indices[0, 35:40])) #1
    MAD = [ int (i) for i in indices[19, 35:40] ] #2
    DR = [ int (i) for i in indices[20, 35:40] ] #3
    LS = [ int (i) for i in indices[21, 35:40] ] #4
    PCA = [ int (i) for i in indices[22, 35:40] ] #5
    NMF = [ int (i) for i in indices[23, 35:40] ] #6

    x = np.arange(len(labels))  # the label locations
    width = 0.14  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - (3*width-width/2), VAR, width, label='VAR')
    rects2 = ax.bar(x - (2*width-width/2), MAD, width, label='MAD')
    rects3 = ax.bar(x - width/2, DR, width, label='DR')
    rects4 = ax.bar(x + width/2, LS, width, label='LS')
    rects5 = ax.bar(x + (2*width-width/2), PCA, width, label='PCA')
    rects6 = ax.bar(x + (3*width-width/2), NMF, width, label='NMF')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Acoustic feature identifier',fontsize='14')
    ax.set_xlabel('Ranking position of the top-five best features',fontsize='14')
    #ax.set_title('YAT3',fontsize='14')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width()/6, height),
                        xytext=(4, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', size=12, rotation=90)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    autolabel(rects6)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=6, fontsize=12)
    fig.tight_layout()
    plt.savefig('D://UNALAcademico//2023//Proyecto//proyectoPTA//unsupervised//Yataros3Morning_barChart_B5.pdf') 
    plt.show()
#----------------------------------------------------------------------------------------------------------------------------------------#
def CorrelationYataros3AfternoonBar():
    
    labels = ["5th Best", "4th Best", "3rd Best", "2nd Best",  "Best"]
    
    #x = range(35,40)
    VAR = [ int (i) for i in indices[24, 35:40] ]   #int(list(indices[0, 35:40])) #1
    MAD = [ int (i) for i in indices[25, 35:40] ] #2
    DR = [ int (i) for i in indices[26, 35:40] ] #3
    LS = [ int (i) for i in indices[27, 35:40] ] #4
    PCA = [ int (i) for i in indices[28, 35:40] ] #5
    NMF = [ int (i) for i in indices[29, 35:40] ] #6

    x = np.arange(len(labels))  # the label locations
    width = 0.14  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - (3*width-width/2), VAR, width, label='VAR')
    rects2 = ax.bar(x - (2*width-width/2), MAD, width, label='MAD')
    rects3 = ax.bar(x - width/2, DR, width, label='DR')
    rects4 = ax.bar(x + width/2, LS, width, label='LS')
    rects5 = ax.bar(x + (2*width-width/2), PCA, width, label='PCA')
    rects6 = ax.bar(x + (3*width-width/2), NMF, width, label='NMF')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Acoustic feature identifier',fontsize='14')
    ax.set_xlabel('Ranking position of the top-five best features',fontsize='14')
    #ax.set_title('YAT3',fontsize='14')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width()/6, height),
                        xytext=(4, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', size=12, rotation=90)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    autolabel(rects6)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=6, fontsize=12)
    fig.tight_layout()
    plt.savefig('D://UNALAcademico//2023//Proyecto//proyectoPTA//unsupervised//Yataros3afternoon_barChart_B5.pdf') 
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------#
def CorrelationYataros3EveningBar():
    
    labels = ["5th Best", "4th Best", "3rd Best", "2nd Best",  "Best"]
    
    #x = range(35,40)
    VAR = [ int (i) for i in indices[30, 35:40] ]   #int(list(indices[0, 35:40])) #1
    MAD = [ int (i) for i in indices[31, 35:40] ] #2
    DR = [ int (i) for i in indices[32, 35:40] ] #3
    LS = [ int (i) for i in indices[33, 35:40] ] #4
    PCA = [ int (i) for i in indices[34, 35:40] ] #5
    NMF = [ int (i) for i in indices[35, 35:40] ] #6

    x = np.arange(len(labels))  # the label locations
    width = 0.14  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - (3*width-width/2), VAR, width, label='VAR')
    rects2 = ax.bar(x - (2*width-width/2), MAD, width, label='MAD')
    rects3 = ax.bar(x - width/2, DR, width, label='DR')
    rects4 = ax.bar(x + width/2, LS, width, label='LS')
    rects5 = ax.bar(x + (2*width-width/2), PCA, width, label='PCA')
    rects6 = ax.bar(x + (3*width-width/2), NMF, width, label='NMF')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Acoustic feature identifier',fontsize='14')
    ax.set_xlabel('Ranking position of the top-five best features',fontsize='14')
    #ax.set_title('YAT3',fontsize='14')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width()/6, height),
                        xytext=(4, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', size=12, rotation=90)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    autolabel(rects6)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=6, fontsize=12)
    fig.tight_layout()
    plt.savefig('D://UNALAcademico//2023//Proyecto//proyectoPTA//unsupervised//Yataros3Evening_barChart_B5.pdf') 
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------#
def CorrelationYataros3AfterMidnightBar():
    
    labels = ["5th Best", "4th Best", "3rd Best", "2nd Best",  "Best"]
    
    #x = range(35,40)
    VAR = [ int (i) for i in indices[36, 35:40] ]   #int(list(indices[0, 35:40])) #1
    MAD = [ int (i) for i in indices[37, 35:40] ] #2
    DR = [ int (i) for i in indices[38, 35:40] ] #3
    LS = [ int (i) for i in indices[39, 35:40] ] #4
    PCA = [ int (i) for i in indices[40, 35:40] ] #5
    NMF = [ int (i) for i in indices[41, 35:40] ] #6

    x = np.arange(len(labels))  # the label locations
    width = 0.14  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - (3*width-width/2), VAR, width, label='VAR')
    rects2 = ax.bar(x - (2*width-width/2), MAD, width, label='MAD')
    rects3 = ax.bar(x - width/2, DR, width, label='DR')
    rects4 = ax.bar(x + width/2, LS, width, label='LS')
    rects5 = ax.bar(x + (2*width-width/2), PCA, width, label='PCA')
    rects6 = ax.bar(x + (3*width-width/2), NMF, width, label='NMF')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Acoustic feature identifier',fontsize='14')
    ax.set_xlabel('Ranking position of the top-five best features',fontsize='14')
    #ax.set_title('YAT3',fontsize='14')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width()/6, height),
                        xytext=(4, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', size=12, rotation=90)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    autolabel(rects6)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=6, fontsize=12)
    fig.tight_layout()
    plt.savefig('D://UNALAcademico//2023//Proyecto//proyectoPTA//unsupervised//Yataros3AfterMidnight_barChart_B5.pdf') 
    plt.show()

   
#----------------------------------------------------------------------------------------------------------------------#
def CorrelationsAllDataSets(df):
    '''plot_color_gradients('Sequential', ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])'''
    df1=df.iloc[[35,36,37,38,39]]
    print("DF1",df1)
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.figure(figsize=(15,40))
    print(type(df1.corr(method='pearson')))
    dataframedf = abs(df1.corr(method='pearson'))
    print(dataframedf)
    # plotting correlation heatmap
    #dataplot = sns.heatmap(df1.corr(), cmap="YlGnBu", annot=False)
    
    dataplot = sns.heatmap(abs(df1.corr(method='pearson')), cmap="OrRd", annot=True,  fmt=".1f",  linewidth=.5)
    
    
    plt.title('Correlación de Pearson  Yataros 1 Todos los Conjuntos de Datos y Todos los métodos')
    #plt.rcParams.update({'font.size': 5})
    #plt.rc('font', size=SMALL_SIZE)     
    plt.show()
    
    
#----------------------------------------------------------------------------------------------------------------------#
def TransformarPosiciones(indices):
    print("Todos Datos")
    print(indices,"", indices.shape)
    
    indicesT = np.zeros(indices.shape)
    
    for i in range(indices.shape[0]):
         fila = indices[i,:]
         for j in range(len(fila)):
             indicesT[i,j] =  list(fila).index(j)
    print("TEMP")
    print(indicesT)
    return indicesT
#----------------------------------------------------------------------------------------------------------------------#
def todosDatos():
    
    etiquetas=['0-ACImain', 
               '1-ACImin', 
                '2-ACImax', 
                '3-ACImean', 
                '4-ACImedian', 
                '5-ACIstd', 
                '6-ACIvar', 
                '7-ADImain', 
                '8-AEImain', 
                '9-BImain', 
                '10-NDSImain', 
                '11-RMSEmin', 
                '12-RMSEmax', 
                '13-RMSEmean', 
                '14-RMSEmedian', 
                '15-RMSEstd', 
                '16-RMSEvar', 
                '17-SCmin', 
                '18-SCmax', 
                '19-SCmean', 
                '20-SCmedian', 
                '21-SCstd', 
                '22-SCvar', 
                '23-SEmain', 
                '24-TEmain', 
                '25-ZCRmin', 
                '26-ZCRmax', 
                '27-ZCRmean', 
                '28-ZCRmedian', 
                '29-ZCRstd', 
                '30-ZCRvar', 
                '31-WSNR', 
                '32-WSNRact', 
                '33-WSNRcount', 
                '34-WSNRdur', 
                '35-NBmain', 
                '36-ADINRmain', 
                '37-AEINRmain', 
                '38-BINRmain', 
                '39-SENRmain']
    
    etiquetasY= ['VAR  ',
                 'MAD  ',
                 'DR  ',
                 'LS  ',
                 'PCA  ',
                 'NMF  ',
                 'VAR Day ',
                 'MAD Day ',
                 'DR Day ',
                 'LS Day ',
                 'PCA Day ',
                 'NMF Day ', 
                 'VAR Night ',
                 'MAD Night ',
                 'DR Night ',
                 'LS Night ',
                 'PCA Night ',
                 'NMF Night ', 
                 'VAR Morning ',
                 'MAD Morning ',
                 'DR Morning ',
                 'LS Morning ',
                 'PCA Morning ',
                 'NMF Morning ',
                 'VAR Afternoon ',
                 'MAD Afternoon ',
                 'DR Afternoon ',
                 'LS Afternoon ',
                 'PCA Afternoon ',
                 'NMF Afternoon ',
                 'VAR Evening ',
                 'MAD Evening ',
                 'DR Evening ',
                 'LS Evening ',
                 'PCA Evening ',
                 'NMF Evening ',
                 'VAR After midnight ',
                 'MAD After midnight ',
                 'DR After midnight ',
                 'LS After midnight ',
                 'PCA After midnight ',
                 'NMF After midnight ']


    indicesT=  TransformarPosiciones(indices) 
    
 
    fig, ax = plt.subplots(1,1) 
    
    ax.xaxis.set(ticks=np.arange(0.5, len(etiquetas)), ticklabels=etiquetas)
    ax.yaxis.set(ticks=np.arange(0.5, len(etiquetasY)), ticklabels=etiquetasY)
    ax.set_xticklabels(etiquetas, rotation=90, ha='right', minor=False, fontsize=6)
    ax.set_yticklabels(etiquetasY, rotation=0, va='bottom', minor=False, fontsize=6) # Nueva línea para intentar alinear etiquetas de filas
    plt.imshow(indicesT, cmap='hot', vmin=0, vmax=39)
    plt.colorbar()
    plt.title("YAT3 all feature selection methods per dataset")
    plt.savefig("D://UNALAcademico//2023//Proyecto//proyectoPTA//unsupervised//Yataros3.pdf", format='pdf', bbox_inches= 'tight')
   
    #plt.show()
    
    fig, ax = plt.subplots(1,1) 
    #ax.xaxis.set(ticks=np.arange(0.5, len(etiquetas)), ticklabels=etiquetas)
    #ax.set_xticklabels(etiquetas, rotation=90, ha='right', minor=False, fontsize=8)
    
    
    etiquetasY=['VAR', 
                'VAR Day', 
                'VAR Night', 
                'VAR Morning', 
                'VAR Afternoon',
                'VAR Evening', 
                'VAR After midnight',  
                'MAD', 
                'MAD Day', 
                'MAD Night',
                'MAD Morning',
                'MAD Afternoon',
                'MAD Evening',
                'MAD After midnight',
                'DR',
                'DR Day',
                'DR Night',
                'DR Morning',
                'DR Afternoon',
                'DR Evening',
                'DR After midnight',
                'LS',
                'LS Day',
                'LS Night',
                'LS Morning',
                'LS Afternoon',
                'LS Evening',
                'LS After midnight',
                'PCA',
                'PCA Day',
                'PCA Night',
                'PCA Morning',
                'PCA Afternoon',
                'PCA Evening',
                'PCA After midnight',
                'NMF',
                'NMF Day',
                'NMF Night',
                'NMF Morning',
                'NMF Afternoon',
                'NMF Evening',
                'NMF After midnight']
    indicesT=  TransformarPosiciones(indexesResMetodos)  
    
    ax.xaxis.set(ticks=np.arange(0.5, len(etiquetas)), ticklabels=etiquetas)
    ax.yaxis.set(ticks=np.arange(0.5, len(etiquetasY)), ticklabels=etiquetasY)
    ax.set_xticklabels(etiquetas, rotation=90, ha='right', minor=False, fontsize=6)
    ax.set_yticklabels(etiquetasY, rotation=0, va='bottom', minor=False, fontsize=6) # Nueva línea para intentar alinear etiquetas de filas
    plt.imshow(indicesT, cmap='hot', vmin=0, vmax=40)
    plt.colorbar()
    plt.title("YAT3 all datasets per feature selection method")
    
    plt.savefig("D://UNALAcademico//2023//Proyecto//proyectoPTA//unsupervised//Yataros3Methods.pdf", format='pdf', bbox_inches= 'tight')
    #plt.show()

def moda(vec):
    vals, counts = np.unique(vec, return_counts=True)
    mode_value = vals[counts == np.max(counts)]
    #print(indices[:,39])
    print("values = ",vals)
    print("Counts" ,counts)
    print(mode_value)
    
'''vec = [ 7,  9, 24, 24, 29,  6,  4,  8, 14, 26, 10,  5,  7,  9, 15, 32, 31, 10,
  9,  9, 26, 39, 32,  2,  7,  9, 24, 31, 10,  6,  7,  8, 24, 24, 23,  6,
  3,  8, 23, 18,  8,  4, 9, 12, 15, 31, 20,  8,  9, 14, 14, 36, 14, 13,  8,  9, 14, 36, 32, 10,
  8,  9, 24, 22, 25,  7,  9, 16, 25, 35, 11, 12,  9, 11, 24, 36, 26,  8,  8,  8, 25, 27, 27,  9,  8, 10, 15, 15, 20,  3,  7, 11, 14, 28,  8,  6,  8,  9, 14, 10, 20,  7,
  7,  8, 25,  7, 30,  5,  8, 11, 23, 26,  8,  6,  7,  9, 23, 26, 11,  5,
  8,  8, 24, 21,  6, 11]'''

print("Ultimos 5 YAT3")
vecY1= indices[0:6, 35:40]
print(vecY1.flatten())
moda(vecY1)
print("---------------------------------")
#vecDia= indices[6:12, 35:40]
#print(vecDia.flatten())
#vecNoche= indices[12:18, 35:40]
#print(vecNoche.flatten())
#vecMorning= indices[18:24, 35:40]
#print(vecMorning.flatten())
#vecAfternoon= indices[24:30, 35:40]
#print(vecAfternoon.flatten())
#vecEvening= indices[30:36, 35:40]
#print(vecEvening.flatten())
#vecAMN= indices[36:42, 35:40]
#print(vecAMN.flatten())
    
#moda(vecY1)

#Generate heat maps
procesarPorMetodos()
todosDatos()   
#Generate bar plots
CorrelationYataros3Bar()
CorrelationYataros3DiaBar()
CorrelationYataros3NocheBar()
CorrelationYataros3MorningBar()
CorrelationYataros3AfternoonBar()
CorrelationYataros3EveningBar()
CorrelationYataros3AfterMidnightBar()
