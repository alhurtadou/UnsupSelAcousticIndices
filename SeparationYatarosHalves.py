import pandas as pd
def yataros():
    filename  = "yataros1.csv"
    filename2 = "yataros2.csv"
    filename3 = "yataros3.csv"
    indexesFile = pd.read_csv(filename)
    indexesFile_top = list(indexesFile.columns) 
    indexesFile_top.pop(0)
    
    indexesFile2 = pd.read_csv(filename2)
    indexesFile_top2 = list(indexesFile2.columns) 
    indexesFile_top2.pop(0)
    
    indexesFile3 = pd.read_csv(filename3)
    indexesFile_top3 = list(indexesFile3.columns) 
    indexesFile_top3.pop(0)
    
    print("Encabezados")
    print(indexesFile_top)
    
    lista = list(indexesFile['filename'])
    print(type(int(lista[3][9:11])))
    posnoche = []
    posdia = []
    for i in range(len(lista)):
        entero = int(lista[i][9:11])
        if  entero > 17 or entero < 6  :
            posnoche.append(i)
        else: 
            posdia.append(i)


    subsetnoche = indexesFile.loc[posnoche]
    subsetdia = indexesFile.loc[posdia]
    print(subsetnoche)
    print(subsetdia)
    subsetnoche.to_csv('Yataros1Noche.csv', index=False)
    subsetdia.to_csv('Yataros1Dia.csv', index=False)

    #####################################################
    lista = list(indexesFile2['filename'])
    print(type(int(lista[3][9:11])))
    posnoche = []
    posdia = []
    for i in range(len(lista)):
        entero = int(lista[i][9:11])
        if  entero > 17 or entero < 6  :
            posnoche.append(i)
        else: 
            posdia.append(i)
    subsetnoche = indexesFile2.loc[posnoche]
    subsetdia = indexesFile2.loc[posdia]
    print(subsetnoche)
    print(subsetdia)
    subsetnoche.to_csv('Yataros2Noche.csv', index=False)
    subsetdia.to_csv('Yataros2Dia.csv', index=False)
    ###################################################################
    lista = list(indexesFile3['filename'])
    print(type(int(lista[3][9:11])))
    posnoche = []
    posdia = []
    for i in range(len(lista)):
        entero = int(lista[i][9:11])
        if  entero > 17 or entero < 6  :
            posnoche.append(i)
        else: 
            posdia.append(i)
    subsetnoche = indexesFile3.loc[posnoche]
    subsetdia = indexesFile3.loc[posdia]
    print(subsetnoche)
    print(subsetdia)
    subsetnoche.to_csv('Yataros3Noche.csv', index=False)
    subsetdia.to_csv('Yataros3Dia.csv', index=False)
yataros()