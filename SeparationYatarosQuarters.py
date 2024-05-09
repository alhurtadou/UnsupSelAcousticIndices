import pandas as pd
filename  = "yataros1.csv"
indexesFile = pd.read_csv(filename)
indexesFile_top = list(indexesFile.columns) 
indexesFile_top.pop(0)

filename2 = "yataros2.csv"   
indexesFile2 = pd.read_csv(filename2)
indexesFile_top2 = list(indexesFile2.columns) 
indexesFile_top2.pop(0)

filename3 = "yataros3.csv"    
indexesFile3 = pd.read_csv(filename3)
indexesFile_top3 = list(indexesFile3.columns) 
indexesFile_top3.pop(0)

def yataros1():
    #print("Encabezados")
    #print(indexesFile_top)
    
    lista = list(indexesFile['filename'])
    posdia = []
    postarde = []
    posnoche = []
    posamanecer = []
    for i in range(len(lista)):
        entero = int(lista[i][9:11])
        print('Entero=', entero, 'i=', i)       
        if  entero >= 6 and entero < 12:
            posdia.append(i)
        elif entero >= 12 and entero < 18:
            postarde.append(i)
        elif entero >= 18 and entero < 24:
            posnoche.append(i)
        else: 
            posamanecer.append(i)

    subsetnoche = indexesFile.loc[posnoche]
    subsetdia = indexesFile.loc[posdia]
    subsettarde = indexesFile.loc[postarde]
    subsetamanecer = indexesFile.loc[posamanecer]
    print('Dia= \n', subsetdia)
    print('Tarde= \n',subsettarde)
    print('Noche= \n',subsetnoche)
    print('Amanecer= \n',subsetamanecer)
    subsetnoche.to_csv('Yataros1Manana.csv', index=False)
    subsetdia.to_csv('Yataros1Tarde.csv', index=False)
    subsettarde.to_csv('Yataros1Anochecer.csv', index=False)
    subsetamanecer.to_csv('Yataros1Amanecer.csv', index=False)
yataros1()

#####################################################
def yataros2():
    lista = list(indexesFile2['filename'])
    print(type(int(lista[3][9:11])))
    posdia = []
    postarde = []
    posnoche = []
    posamanecer = []
    for i in range(len(lista)):
        entero = int(lista[i][9:11])
        print('Entero=', entero, 'i=', i)       
        if  entero >= 6 and entero < 12:
            posdia.append(i)
        elif entero >= 12 and entero < 18:
            postarde.append(i)
        elif entero >= 18 and entero < 24:
            posnoche.append(i)
        else: 
            posamanecer.append(i)


    subsetnoche = indexesFile2.loc[posnoche]
    subsetdia = indexesFile2.loc[posdia]
    subsettarde = indexesFile2.loc[postarde]
    subsetamanecer = indexesFile2.loc[posamanecer]
    print('Dia= \n', subsetdia)
    print('Tarde= \n',subsettarde)
    print('Noche= \n',subsetnoche)
    print('Amanecer= \n',subsetamanecer)
    subsetnoche.to_csv('Yataros2Manana.csv', index=False)
    subsetdia.to_csv('Yataros2Tarde.csv', index=False)
    subsettarde.to_csv('Yataros2Anochecer.csv', index=False)
    subsetamanecer.to_csv('Yataros2Amanecer.csv', index=False)
yataros2()
    ###################################################################
def yataros3():
    lista = list(indexesFile3['filename'])
    print(type(int(lista[3][9:11])))
    posdia = []
    postarde = []
    posnoche = []
    posamanecer = []
    for i in range(len(lista)):
        entero = int(lista[i][9:11])
        print('Entero=', entero, 'i=', i)       
        if  entero >= 6 and entero < 12:
            posdia.append(i)
        elif entero >= 12 and entero < 18:
            postarde.append(i)
        elif entero >= 18 and entero < 24:
            posnoche.append(i)
        else: 
            posamanecer.append(i)
    subsetnoche = indexesFile3.loc[posnoche]
    subsetdia = indexesFile3.loc[posdia]
    subsettarde = indexesFile3.loc[postarde]
    subsetamanecer = indexesFile3.loc[posamanecer]
    print('Dia= \n', subsetdia)
    print('Tarde= \n',subsettarde)
    print('Noche= \n',subsetnoche)
    print('Amanecer= \n',subsetamanecer)
    subsetnoche.to_csv('Yataros3Manana.csv', index=False)
    subsetdia.to_csv('Yataros3Tarde.csv', index=False)
    subsettarde.to_csv('Yataros3Anochecer.csv', index=False)
    subsetamanecer.to_csv('Yataros3Amanecer.csv', index=False)
yataros3()
