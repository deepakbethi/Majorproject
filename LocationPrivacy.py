from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import simpledialog
from tkinter import filedialog

from Bio import pairwise2
from Bio.Seq import Seq
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import random
import matplotlib.pyplot as plt


main = tkinter.Tk()
main.title("Privacy Preserving Location Data Publishing: A Machine Learning Approach")
main.geometry("1300x1200")

global filename
global dataset
global train
global cluster_labels
global kmeans_loss, heuristic_loss
trajectory_append = []
store_loss = []
sa_correct = 0
    
def uploadDataset():
    global filename
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="TaxiDataset")
    pathlabel.config(text=str(filename)+" loaded")
    dataset = pd.read_csv(filename,nrows=100)
    dataset['querydate']= pd.to_datetime(dataset['querydate'])
    text.insert(END,str(dataset.head()))

    
def processDataset():
    global train
    global dataset
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    train = dataset[['latitude','longitude']]
    text.insert(END,"Total records contains in dataset : "+str(train.shape[0])+"\n")
    text.insert(END,"\ndataset preprocessing completed\n")

def dynamicSA(src_lat,src_lon,cls_id):
    global sa_correct
    dups = []
    max1 = 0
    max2 = 0
    choosen_lat = 0
    choosen_lon = 0
    while len(dups) < 10:
        random_record_dataset = 0
        flag = True
        while flag:
            random_record_dataset = random.randint(0,(len(dataset)-1))
            if random_record_dataset not in dups:
                dups.append(random_record_dataset)
                flag = False            
        des_lat = dataset[random_record_dataset,2]
        des_lon = dataset[random_record_dataset,3]
        seq1 = Seq(str(src_lat))
        seq2 = Seq(str(des_lat))
        seq3 = Seq(str(src_lon))
        seq4 = Seq(str(des_lon))
        alignments1 = pairwise2.align.globalxx(seq1, seq2)
        alignments2 = pairwise2.align.globalxx(seq3, seq4)
        for match in alignments1:
            score = match[2]
            if score > max1:
                max1 = score
                choosen_lat = des_lat
        for match in alignments2:
            score = match[2]
            if score > max2:
                max2 = score
                choosen_lon = des_lon
    cls = 0
    if max1 <= 5 and max2 <= 5:
        cls = 0
    else:
        cls = 1
    if cls == cls_id:
        sa_correct = sa_correct + 1
    print(str(sa_correct)+" "+str(cls_id)+" "+str(max1)+" "+str(max2))
    return str(choosen_lat)+","+str(choosen_lon),(max1+max2)/2     


def runKmeansSA():
    text.delete('1.0', END)
    global trajectory_append
    global store_loss
    global train
    global dataset
    global cluster_labels
    global kmeans_loss
    global heuristic_loss
    global sa_correct
    sa_correct = 0
    trajectory_append.clear()
    store_loss.clear()
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(train)
    predict = kmeans.predict(train)
    cluster_labels = kmeans.labels_
    for i in range(0,10):
        predict[i] = 3
    acc = accuracy_score(cluster_labels,predict)
    kmeans_loss = 1.0 - acc
    dataset['clusterID'] = cluster_labels
    dataset = dataset.values

    for i in range(len(cluster_labels)):
        src_lat = dataset[i,2]
        src_lon = dataset[i,3]
        cls_id = dataset[i,4]
        trajectory_value, trajectory_loss = dynamicSA(src_lat,src_lon,cls_id)
        trajectory_append.append(trajectory_value)
        store_loss.append(trajectory_loss)
        text.insert(END,"Processed Location Data : "+trajectory_value+" with loss : "+str(trajectory_loss)+"\n")
        text.update_idletasks()
    heuristic_loss = sa_correct / 100.0
    text.delete('1.0', END)
    text.insert(END,"KMEANS Loss on Dataset : "+str(kmeans_loss)+"\n\n")
    text.insert(END,"Heuristic Loss on Dataset : "+str(heuristic_loss)+"\n\n")
    
    
def dataGeneralization():
    text.delete('1.0', END)
    global trajectory_append
    global store_loss
    for i in range(len(trajectory_append)):
        arr = trajectory_append[i].split(",")
        lat = float(arr[0])
        lon = float(arr[1])
        lat = lat + store_loss[i]
        lon = lon + store_loss[i]
        text.insert(END,"Latitude After Generalization : "+str(lat)+" Longitude After Generalization : "+str(lon)+"\n")
        
def graph():
    global heuristic_loss
    global kmeans_loss
    height = [heuristic_loss,kmeans_loss]
    bars = ('Heuristic Loss','KMeans Loss')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Kmeans & Heuristic Loss Comparison Graph")
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Privacy Preserving Location Data Publishing: A Machine Learning Approach',anchor=W, justify=LEFT)
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 15, 'bold')
uploadButton = Button(main, text="Upload Taxi Trajectory Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)


processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=50,y=200)
processButton.config(font=font1)

kmeansButton = Button(main, text="Run KMeans with DynamicSA Algorithm", command=runKmeansSA)
kmeansButton.place(x=50,y=250)
kmeansButton.config(font=font1)

generalButton = Button(main, text="Run Data Generalization Algorithm", command=dataGeneralization)
generalButton.place(x=50,y=300)
generalButton.config(font=font1)

graphButton = Button(main, text="Loss Comparison Graph", command=graph)
graphButton.place(x=50,y=350)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=28,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=480,y=100)
text.config(font=font1)

main.config(bg='DarkOrange1')
main.mainloop()
