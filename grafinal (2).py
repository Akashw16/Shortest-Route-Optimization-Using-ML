#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import matplotlib.pyplot as plt
import array as arr
from array import *
from geopy.geocoders import Nominatim 
from geopy import distance 
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import pickle
from geopy.geocoders import Nominatim
from sklearn.model_selection import train_test_split
import datetime
import pytz
pd.set_option('display.max_columns', None)
#libraries import
"""
networks helps to create the connect the points in the graph
matplotlib used for the plotting thee graph
"""


# In[2]:


sample_df = pd.read_csv("D:/beprojectdataset/be_project_train2.csv")


# In[3]:


sample_df.head(1)


# In[ ]:





# In[5]:


a=arr.array( 'd' , [18.46781,18.46415,18.46886,18.47079,18.47447,18.48445,18.57463] )
b=arr.array('d',[73.86823,73.86827,73.86013,73.85764,73.85352,73.7780,73.4563])
#testing locations in thee format of latitude and longitude 
#array a store the latidude and lonngitude is store in the array b


# In[6]:


test_locations = {'L1': (18.46781, 73.86823),
                  'L2': (18.46415, 73.86827),
                  'L3': (18.46886, 73.86013),
                  'L4': (18.47079, 73.85764),
                  'L5': (18.47447, 73.85352),
                  'L6':(18.48445,73.7780),
                  'L7':(18.57463,73.4563),
                  
             }


# In[7]:


val3=arr.array('d',[])
values = test_locations.values()
val3 = list(values)
print(val3)
val3[1]


# In[8]:


#for obtaining the distacee between the every two points(latudude and longitude )
#like 1->2,1->3,1->4,1->5,2->3,.......,4->5
#for that the for loop is prefer and also append method is also used 
#In addition to this distace is calculated using the distace api from distance library
rows, cols = (6, 7)
arr3 = [[0]*cols]*rows
print(arr3)
dis=arr.array('d',[])
for i in range(0,6):
    for j in range(i+1,7):
        dis.append(distance.distance(val3[i],val3[j]).km)
        arr3[i][j]=(distance.distance(val3[i],val3[j]).km)
        print(distance.distance(val3[i],val3[j]).km,"kms")
#print(arr3)


# In[9]:


#
print(dis)


# In[10]:


cordlat=np.array([[a[0]],[a[1]],[a[2]],[a[3]],[a[4]]])
cordlon=np.array([[b[0]],[b[1]],[b[2]],[b[3]],[b[4]]])


# In[11]:


#for the accurate placing the coordinate on graph we resize them(latitude and longitude) with minmax scaler
#by using the minmax scaler the values are converted betweeen 0 to 1
scaler=MinMaxScaler()


# In[12]:


rescaled_cordlat=scaler.fit_transform(cordlat)
rescaled_cordlon=scaler.fit_transform(cordlon)


# In[13]:


#here is the result of that values between 0-1 for latidude
rescaled_cordlat


# In[14]:


#here is the result of that values between 0-1 for longitude
rescaled_cordlon


# In[15]:


#converted values are then store in the seperated array ax for latitude and bx for longitude 
#here array append method is used
ax=arr.array('d',[])
bx=arr.array('d',[])
for  i in range (0,5):
    ax.append(rescaled_cordlat[i])
    bx.append(rescaled_cordlon[i])


# In[16]:


print(ax)
print(bx)


# In[17]:


G=nx.DiGraph()
G.add_weighted_edges_from([('A','B',dis[0]),('A','C',dis[1]),('A','D',dis[2]),('A','E',dis[3]),('B','C',dis[4]),('B','D',dis[5]),('B','E',dis[6]),('C','D',dis[7]),('C','E',dis[8]),('D','E',dis[9])])
labels=nx.get_edge_attributes(G,'weight')


# In[18]:


pos={
    "A":(ax[0],bx[0]),
    "B":(ax[1],bx[1]),
    "C":(ax[2],bx[2]),
    "D":(ax[3],bx[3]),
    "E":(ax[4],bx[4])
}


# In[19]:


#pos = nx.spring_layout(G)
nx.draw_networkx_edges(G,pos=pos,edgelist=G.edges(),edge_color='black')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
nx.draw_networkx_nodes(G,pos,node_size=500)
nx.draw_networkx_labels(G,pos)
plt.show()


# In[20]:


#creating  a graph 
import matplotlib.pyplot as plt
import networkx as nx
G=nx.Graph()

#node names are given as abcde
G.add_nodes_from("ABCDE")
#setting a edges from where to where like AC means edge from A to C
"AB","AD","AE","BE","CE"
G.add_edges_from(["BC","BD","BE","CD","CE","DE"])


# In[21]:


#1
#G.add_nodes_from("ABCDE")
#setting a edges from where to where like AC means edge from A to C

#G.add_edges_from(["AC","AB","AD","AE"])


# In[22]:


pos={
    "A":(ax[0],bx[0]),
    "B":(ax[1],bx[1]),
    "C":(ax[2],bx[2]),
    "D":(ax[3],bx[3]),
    "E":(ax[4],bx[4])
}


# In[23]:


#FOR PLOTTING THEM IN THE ACCURATE FORMAT ONTHE GRAAPH WE USED THEE POS(POSITION) ATTRIBUTE OF THE NETWORKS LIBRARY
pos={
    "A":(ax[0],bx[0]),
    "B":(ax[1],bx[1]),
    "C":(ax[2],bx[2]),
    "D":(ax[3],bx[3]),
    "E":(ax[4],bx[4])
}


# In[24]:


nx.draw(G,pos=pos,with_labels=True,
        node_color="red",node_size=3000,
        font_color="white",font_size=20,font_family="Times New Roman", font_weight="bold",
        width=5)
plt.margins(0.2)
plt.show()


# In[ ]:





# In[25]:


ps={
    "A":(ax[0],bx[0]),
    "B":(ax[1],bx[1]),
    "C":(ax[2],bx[2]),
    "D":(ax[3],bx[3]),
    "E":(ax[4],bx[4])
}


# In[26]:


import matplotlib.pyplot as plt
import networkx as nx
G=nx.Graph()

#node names are given as abcde
G.add_nodes_from("ABCDE")
#setting a edges from where to where like AC means edge from A to C
"AB","AD","AE","BE","CE"
G.add_edges_from(["BC"])


# In[27]:


nx.draw(G,pos=pos,with_labels=True,
        node_color="red",node_size=3000,
        font_color="white",font_size=20,font_family="Times New Roman", font_weight="bold",
        width=5)
plt.margins(0.2)
plt.show()


# In[28]:


G.add_nodes_from("ABCDE")
#setting a edges from where to where like AC means edge from A to C
G.add_edges_from(["BC","AC","AD","AE"])


# In[29]:


nx.draw(G,pos=ps,with_labels=True,
        node_color="red",node_size=3000,
        font_color="white",font_size=20,font_family="Times New Roman", font_weight="bold",
        width=5)
plt.margins(0.2)
plt.show()


# In[30]:


plt.plot(rescaled_cordlat,rescaled_cordlon)


# In[31]:


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.set_title('plot gragh',fontsize = 24)

ax.set_xlabel('rescaled_cordlat',fontsize = 24)
ax.set_ylabel('rescaled_cordlon',fontsize = 24)
#plt.plot(rescaled_cordlat,rescaled_cordlon,'-ok')
##plt.xlim(0, 1)
##plt.ylim(0, 1)
plt.plot(rescaled_cordlat,rescaled_cordlon, marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")
G=nx.DiGraph()

#G.add_weighted_edges_from([(0,1,dis[0]),(0,2,dis[1]),('a','d',dis[2]),('a','e',dis[3]),('b','c',dis[4]),('b','d',dis[5]),('b','e',dis[6]),('c','d',dis[7]),('c','e',dis[8]),('d','e',dis[9])])
labels=nx.get_edge_attributes(G,'weight')

pos = nx.spring_layout(G)
nx.draw_networkx_edges(G,pos,edgelist=G.edges(),edge_color='black')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
nx.draw_networkx_nodes(G,pos,node_size=500)
nx.draw_networkx_labels(G,pos)
plt.show()


# In[32]:


#importing direct from the datasets
#sample_df = pd.read_csv("D:/beprojectdataset/aakash/Delivery_truck_trip_data4.csv")


# In[33]:


sample_df.head(5)


# In[34]:


sample_df.pivot_table(index="pickup_latitude").head(5)


# In[35]:


sample_df.pickup_latitude[0:10]
sample_df.pickup_longitude[0:10]
sample_df.values[0][2]


# In[36]:


forlat=arr.array('d',[])
forlon=arr.array('d',[])
for i in range (0,10):
    forlat.append(sample_df.values[i][1])
    forlon.append(sample_df.values[i][2])


# In[37]:


print(forlat)
print(forlon)
sx1=len(forlat)
sx2=len(forlon)
print(sx1)
print(sx2)


# In[38]:


for i in range (0,sx1):
    merged_list = [(forlat[i], forlon[i]) for i in range(0, sx1)]
merged_list


# In[39]:


distforall=arr.array('d',[])
for i in range(0,sx1-1):
    for j in range(i+1,sx1):
        distforall.append(distance.distance(merged_list[i],merged_list[j]).km)
        #print(distance.distance(merged_list[i],merged_list[j]).km,"kms")


# In[40]:


#print(distforall)


# In[41]:


scaler1=MinMaxScaler()


# In[42]:


cordlat1=np.array('d',[])
cordlon1=np.array('d',[])
for i in range (0,sx1):
    cordlat1=np.array([[forlat[i]]])
    cordlon1=np.array([[forlon[i]]])


# In[43]:


rescaled_cordlat2=scaler1.fit_transform(cordlat1)
rescaled_cordlon2=scaler1.fit_transform(cordlon1)
rescaled_cordlat2


# In[44]:


scaler2 = MinMaxScaler()
model=scaler2.fit(merged_list)
scaled_data=model.transform(merged_list)


# In[45]:


print(scaled_data)


# In[46]:


scaled_data[3][0]


# In[47]:


ax1=arr.array('d',[])
bx1=arr.array('d',[])
for  i in range (0,sx1):
    ax1.append(scaled_data[i][0])
    bx1.append(scaled_data[i][1])

bx1


# In[48]:


import matplotlib.pyplot as plt
import networkx as nx
G=nx.Graph()
nodes=arr.array('i',[])
for i in range(0,sx1):
    nodes.append(i)
nodes
#for i in range(0,sx1):
 #   G.add_nodes_from(nodes)
  #  G.add_edges_from(["])
#node names are given as abcde
#G.add_nodes_from("ABCDE")
#setting a edges from where to where like AC means edge from A to C
#G.add_edges_from(["AC","BC","BD","CD","DE","AB","AD","AE","BE","CE"])


# In[49]:


plt.plot(ax1,bx1)


# In[51]:


#poss={
#    "1":(ax1[0],bx1[0]),"2":(ax1[1],bx1[1]),"3":(ax1[2],bx1[2]),"4":(ax1[3],bx1[3]),"5":(ax1[4],bx1[4]),"6":(ax1[5],bx1[5]),"7":(ax1[6],bx1[6]),
#    "8":(ax1[7],bx1[7]),"9":(ax1[8],bx1[8]),"10":(ax1[9],bx1[9]),"11":(ax1[10],bx1[10]),"12":(ax1[11],bx1[11]),"13":(ax1[12],bx1[12]),"14":(ax1[13],bx1[13]),
#    "15":(ax1[14],bx1[14]),"16":(ax1[15],bx1[15]),"17":(ax1[16],bx1[16]),"18":(ax1[17],bx1[17]),"19":(ax1[18],bx1[18]),"20":(ax1[19],bx1[19]),"21":(ax1[20],bx1[20]),
 #   "22":(ax1[21],bx1[21]),"23":(ax1[22],bx1[22]),"24":(ax1[23],bx1[23]),"25":(ax1[24],bx1[24]),"26":(ax1[25],bx1[25]),"27":(ax1[26],bx1[26]),"28":(ax1[27],bx1[27]),
#    "29":(ax1[28],bx1[28]),"30":(ax1[29],bx1[29]),"31":(ax1[30],bx1[30]),"32":(ax1[31],bx1[31]),"33":(ax1[32],bx1[32]),"34":(ax1[33],bx1[33])
    
#}
p0s1={
    "A":(ax[0],bx[0]),
    "B":(ax[1],bx[1]),
    "C":(ax[2],bx[2]),
    "D":(ax[3],bx[3]),
    "E":(ax[4],bx[4])
}


# In[ ]:


fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.set_title('plot gragh',fontsize = 24)

ax.set_xlabel('rescaled_cordlat2',fontsize = 24)
ax.set_ylabel('rescaled_cordlon2',fontsize = 24)
#plt.plot(rescaled_cordlat,rescaled_cordlon,'-ok')
##plt.xlim(0, 1)
##plt.ylim(0, 1)
plt.plot(ax1,bx1, marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")
G=nx.DiGraph()

#G.add_weighted_edges_from([(0,1,dis[0]),(0,2,dis[1]),('a','d',dis[2]),('a','e',dis[3]),('b','c',dis[4]),('b','d',dis[5]),('b','e',dis[6]),('c','d',dis[7]),('c','e',dis[8]),('d','e',dis[9])])
labels=nx.get_edge_attributes(G,'weight')


# In[ ]:


G=nx.DiGraph()
for i in range (0,sx1-1):
    for j in range(i+1,sx1):
        G.add_weighted_edges_from([(nodes[i],nodes[j],distforall[i])])
        labels=nx.get_edge_attributes(G,'weight')


# In[ ]:


nx.draw(G,pos,with_labels=True,
        node_color="red",node_size=3000,
        font_color="white",font_size=20,font_family="Times New Roman", font_weight="bold",
        width=5)
plt.margins(0.2)
plt.show()


# In[52]:


import matplotlib.pyplot as plt
import networkx as nx

G=nx.Graph()

G = nx.DiGraph()


# In[53]:


#G.add_nodes_from(["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23",
#                 "24","25","26","27","28","29","30","31","32","33","34"])


# In[54]:


#pos1=[]
#pos1= [(nodes[i], nodes[j]) for i in range(0, sx1-1) for j in range(i+1,sx1)]
#pos1


# In[55]:


#for i in range(0,len(pos1)):
#    G.add_edges_from([pos1])


# In[56]:


pos2={
    "1":(ax1[0],bx1[0]),"2":(ax1[1],bx1[1]),"3":(ax1[2],bx1[2]),"4":(ax1[3],bx1[3]),"5":(ax1[4],bx1[4]),"6":(ax1[5],bx1[5]),"7":(ax1[6],bx1[6]),
    "8":(ax1[7],bx1[7]),"9":(ax1[8],bx1[8]),"10":(ax1[9],bx1[9]),"11":(ax1[10],bx1[10]),"12":(ax1[11],bx1[11]),"13":(ax1[12],bx1[12]),"14":(ax1[13],bx1[13]),
    "15":(ax1[14],bx1[14]),"16":(ax1[15],bx1[15]),"17":(ax1[16],bx1[16]),"18":(ax1[17],bx1[17]),"19":(ax1[18],bx1[18]),"20":(ax1[19],bx1[19]),"21":(ax1[20],bx1[20]),
    "22":(ax1[21],bx1[21]),"23":(ax1[22],bx1[22]),"24":(ax1[23],bx1[23]),"25":(ax1[24],bx1[24]),"26":(ax1[25],bx1[25]),"27":(ax1[26],bx1[26]),"28":(ax1[27],bx1[27]),
    "29":(ax1[28],bx1[28]),"30":(ax1[29],bx1[29]),"31":(ax1[30],bx1[30]),"32":(ax1[31],bx1[31]),"33":(ax1[32],bx1[32]),"34":(ax1[33],bx1[33])
    
}


# In[57]:


pos2 = nx.spring_layout(G)
nx.draw_networkx_edges(G,pos2,edgelist=G.edges(),edge_color='black')
nx.draw_networkx_edge_labels(G,pos2,edge_labels=labels)
nx.draw_networkx_nodes(G,pos2,node_size=500)
nx.draw_networkx_labels(G,pos2)
plt.show()


# In[ ]:





# In[50]:


INF=9999999
N=5
G = [
    [0,dis[0],dis[1],dis[2],dis[3]],
    [dis[0],0,dis[4],dis[5],dis[6]],
    [dis[1],dis[4],0,dis[7],dis[8]],
    [dis[2],dis[5],dis[7],0,dis[9]],
    [dis[3],dis[6],dis[8],dis[9],0]]
selected_node=[0,0,0,0,0]
no_node=0
selected_node[0]=True
print("edge : weight\n")
while(no_node<N-1):
    minimum=INF
    a=0
    b=0
    for m in range(N):
        if selected_node[m]:
            for n in range(N):
                if((not selected_node[n]) and G[m][n]):
                    if minimum > G[m][n]:
                        minimum=G[m][n]
                        a=m
                        b=n
    print(str(a)+" - "+str(b)+" :"+str(G[a][b]))
    selected_node[b]=True
    no_node+=1
        


# In[51]:


rows, cols = (4, 5)
arr1 = [[0]*cols]*rows
print(arr1)
dis


# In[52]:


k=0
for i in range (0,4):
    for j in range(i+1,5):
        arr1[i][j]=dis[k]
        k=k+1;
print(arr1)


# In[65]:



H = [
    [0,1,35,60,4,7,6],
    [1,0,5,6,7,5,10],
    [2,5,0,18,9,35,45,],
    [3,6,8,0,10,11,12],
    [4,7,9,10,0,3,2],
    [3,6,8,0,10,11,12],
    [4,7,9,10,0,3,2]]

I=[[0, 1, 6, 7, 4, 6, 6],
 [1, 0, 5, 5, 5, 5, 7],
 [2, 5, 0, 9, 6, 8, 8],
 [3, 6, 8, 0, 7, 9, 9],
 [4, 7, 9, 10, 0, 3, 2],
 [3, 6, 8, 0, 10, 11, 9],
 [4, 7, 9, 10, 0, 3, 2]]


# In[62]:


for i in range (0,6):
    for j in range(i+1,7):
        print("{} ---> {}   {}".format(i+1,j+1,H[i][j]))


# In[63]:


#2nd stage of distance for three points#H
for i in range (0,6):
    for j in range(i+1,7):
        for k in range(0,7):
            if(k!=i or k!=j):
                if(H[i][k]+H[k][j]<H[i][j]):
                    H[i][j]=H[i][k]+H[k][j]
                    print("{} ---> {}----> {}   {}".format(i+1,k+1,j+1,H[i][j]))
                else:
                    continue


# In[64]:


H


# In[74]:


#3stage with 


# In[66]:


for i in range (0,6):
    for j in range(i+1,7):
        for k in range(0,7):
            if(k!=i and k!=j):
                for l in range (0,7):
                    if(l!=k ):
                        if(I[i][k]+I[k][l]+I[l][j]<I[i][j]):
                            I[i][j]= I[i][k]+I[k][l]+I[l][j]
                            print("{} ---> {}----> {}---->{}   {}".format(i+1,k+1,l+1,j+1,I[i][j]))
                        else:
                            continue
                        


# In[67]:


I


# In[60]:


#5th rotation


# In[69]:


for i in range (0,6):
    for j in range(i+1,7):
        for k in range(0,7):
            if(k!=i or k!=j):
                for l in range (0,7):
                    if(l!=i or l!=j or l!=k):
                        for m in range(0,7):
                             if(m!=i or m!=j or m!=k or m!=l):
                                    if(I[i][k]+I[k][l]+I[l][m]+I[m][j]<I[i][j]):
                                        I[i][j]= I[i][k]+I[k][l]+I[l][m]+i[m][j]
                                        print("{} ---> {}----> {}---->{}----{}  {}".format(i+1,k+1,l+1,m+1,j+1,I[i][j]))
                                    else:
                                        continue


# In[70]:


I


# In[71]:


for i in range (0,4):
    for j in range(i+1,5):
        for k in range(0,5):
            if(k!=i or k!=j):
                for l in range (0,5):
                    if(l!=i or l!=j or l!=k):
                        for m in range(0,5):
                             if(m!=i or m!=j or m!=k):
                                    for n in range(0,5):
                                        if(n!=i or n!=j or n!=k or n!=l or n!=m):
                                             if(I[i][k]+I[k][l]+I[l][m]+I[m][n]+I[n][j]<I[i][j]):
                                                    I[i][j]= I[i][k]+I[k][l]+I[l][m]+I[m][n]+i[n][j]
                                                    print("{} ---> {}----> {}---->{}----{}----{}  {}".format(i+1,k+1,l+1,m+1,n+1,j+1,I[i][j]))
                                                    
                        


# In[72]:


#6th iteration 


# In[73]:


for i in range (0,4):
    for j in range(i+1,5):
        for k in range(0,5):
            if(k!=i or k!=j):
                for l in range (0,5):
                    if(l!=i or l!=j or l!=k):
                        for m in range(0,5):
                             if(m!=i or m!=j or m!=k):
                                    for n in range(0,5):
                                        if(n!=i or n!=j or n!=k or n!=l or n!=m):
                                            for o in range(0,5):
                                                if(o!=i or o!=j or o!=k or o!=l or o!=m or o!=n):
                                                    if(I[i][k]+I[k][l]+I[l][m]+I[m][n]+I[n][o]+I[o][j]<I[i][j]):
                                                        I[i][j]= I[i][k]+I[k][l]+I[l][m]+I[m][n]+I[n][o]+I[o][j]
                                                        print("{} ---> {}----> {}---->{}----{}----{}--{} {}".format(i+1,k+1,l+1,m+1,n+1,o+1,j+1,I[i][j]))
                                                    
                        


# In[74]:


#7th rotaion 


# In[75]:


for i in range (0,4):
    for j in range(i+1,5):
        for k in range(0,5):
            if(k!=i or k!=j):
                for l in range (0,5):
                    if(l!=i or l!=j or l!=k):
                        for m in range(0,5):
                             if(m!=i or m!=j or m!=k):
                                    for n in range(0,5):
                                        if(n!=i or n!=j or n!=k or n!=l or n!=m):
                                            for o in range(0,5):
                                                if(o!=i or o!=j or o!=k or o!=l or o!=m or o!=n):
                                                    for p in range(0,5):
                                                        if(p!=i or p!=j or p!=k or p!=l or p!=m or p!=n or p!=o):
                                                            if(I[i][k]+I[k][l]+I[l][m]+I[m][n]+I[n][o]+I[o][p]+I[p][j]<I[i][j]):
                                                                I[i][j]= I[i][k]+I[k][l]+I[l][m]+I[m][n]+I[n][o]+I[o][p]+I[p][j]
                                                                print("{} ---> {}----> {}---->{}----{}----{}--{}--{} {}".format(i+1,k+1,l+1,m+1,n+1,o+1,p+1,j+1,I[i][j]))
                                                    
                        


# In[76]:


#8 th rotaion 


# In[77]:


for i in range (0,4):
    for j in range(i+1,5):
        for k in range(0,5):
            if(k!=i or k!=j):
                for l in range (0,5):
                    if(l!=i or l!=j or l!=k):
                        for m in range(0,5):
                             if(m!=i or m!=j or m!=k):
                                    for n in range(0,5):
                                        if(n!=i or n!=j or n!=k or n!=l or n!=m):
                                            for o in range(0,5):
                                                if(o!=i or o!=j or o!=k or o!=l or o!=m or o!=n):
                                                    for p in range(0,5):
                                                        if(p!=i or p!=j or p!=k or p!=l or p!=m or p!=n or p!=o):
                                                            for q in range(0,5):
                                                                if(q!=i or q!=j or q!=k or q!=l or q!=m or q!=n or q!=o or q!=p):
                                                                    if(I[i][k]+I[k][l]+I[l][m]+I[m][n]+I[n][o]+I[o][p]+I[p][q]+I[q][j]<I[i][j]):
                                                                        I[i][j]= I[i][k]+I[k][l]+I[l][m]+I[m][n]+I[n][o]+I[o][p]+I[p][q]+I[q][j]
                                                                        print("{} ---> {}----> {}---->{}----{}----{}--{}--{}---{} {}".format(i+1,k+1,l+1,m+1,n+1,o+1,p+1,q+1,j+1,I[i][j]))
                                                    
                        


# In[78]:


#9th rotation


# In[79]:


for i in range (0,4):
    for j in range(i+1,5):
        for k in range(0,5):
            if(k!=i or k!=j):
                for l in range (0,5):
                    if(l!=i or l!=j or l!=k):
                        for m in range(0,5):
                             if(m!=i or m!=j or m!=k):
                                    for n in range(0,5):
                                        if(n!=i or n!=j or n!=k or n!=l or n!=m):
                                            for o in range(0,5):
                                                if(o!=i or o!=j or o!=k or o!=l or o!=m or o!=n):
                                                    for p in range(0,5):
                                                        if(p!=i or p!=j or p!=k or p!=l or p!=m or p!=n or p!=o):
                                                            for q in range(0,5):
                                                                if(q!=i or q!=j or q!=k or q!=l or q!=m or q!=n or q!=o or q!=p):
                                                                    for r in range(0,5):
                                                                        if(r!=i or r!=j or r!=k or r!=l or r!=m or r!=n or r!=o or r!=p or r!=q):
                                                                            if(I[i][k]+I[k][l]+I[l][m]+I[m][n]+I[n][o]+I[o][p]+I[p][q]+I[q][r]+I[r][j]<I[i][j]):
                                                                                I[i][j]= I[i][k]+I[k][l]+I[l][m]+I[m][n]+I[n][o]+I[o][p]+I[p][q]+I[q][r]+I[r][j]
                                                                                print("{} ---> {}----> {}---->{}----{}----{}--{}--{}---{}---{} {}".format(i+1,k+1,l+1,m+1,n+1,o+1,p+1,q+1,r+1,j+1,I[i][j]))
                                                    
                        


# In[80]:


#10th rotaion


# In[81]:


for i in range (0,4):
    for j in range(i+1,5):
        for k in range(0,5):
            if(k!=i or k!=j):
                for l in range (0,5):
                    if(l!=i or l!=j or l!=k):
                        for m in range(0,5):
                             if(m!=i or m!=j or m!=k):
                                    for n in range(0,5):
                                        if(n!=i or n!=j or n!=k or n!=l or n!=m):
                                            for o in range(0,5):
                                                if(o!=i or o!=j or o!=k or o!=l or o!=m or o!=n):
                                                    for p in range(0,5):
                                                        if(p!=i or p!=j or p!=k or p!=l or p!=m or p!=n or p!=o):
                                                            for q in range(0,5):
                                                                if(q!=i or q!=j or q!=k or q!=l or q!=m or q!=n or q!=o or q!=p):
                                                                    for r in range(0,5):
                                                                        if(r!=i or r!=j or r!=k or r!=l or r!=m or r!=n or r!=o or r!=p or r!=q):
                                                                            for s in range(0,5):
                                                                                if(s!=i or s!=j or s!=k or s!=l or s!=m or s!=n or s!=o or s!=p or s!=q or s!=r):
                                                                                    if(I[i][k]+I[k][l]+I[l][m]+I[m][n]+I[n][o]+I[o][p]+I[p][q]+I[q][r]+I[r][s]+I[s][j]<I[i][j]):
                                                                                        I[i][j]= I[i][k]+I[k][l]+I[l][m]+I[m][n]+I[n][o]+I[o][p]+I[p][q]+I[q][r]+I[r][s]+I[s][j]
                                                                                        print("{} ---> {}----> {}---->{}----{}----{}--{}--{}---{}---{}---{} {}".format(i+1,k+1,l+1,m+1,n+1,o+1,p+1,q+1,r+1,s+1,j+1,I[i][j]))
                                                    
                        


# In[82]:


print("source   destination   PATH DISTANCE")
for i in range (0,4):
    
    for j in range(i+1,5):
        
        print("{}    --->    {}             {}".format(i+1,j+1,I[i][j]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




