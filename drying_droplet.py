import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import pandas as pd
from plantcv import plantcv as pcv
from skimage.draw import disk


file_list = glob.glob(r"C:\Users\Nishanth\Desktop\drying droplets/*.*") 
print(file_list) 

df_final= pd.DataFrame()
final_Dia= pd.DataFrame()
final_Vol= pd.DataFrame()
final_Concentration= pd.DataFrame()
final= pd.DataFrame()



kernal = np.ones((2,2),np.uint8)
ret=0
l=0
my_list=[]
y_min=[float('nan')]
x_min=[float('nan')]
y_max=[float('nan')]
x_max=[float('nan')]
pixcon=0.6536
h=25
c=7



path = r"C:\Users\Nishanth\Desktop\drying droplets/*.*"
for file in glob.glob(path):   
    print(file)     
    a= cv2.imread(file,0)
    a=cv2.GaussianBlur(a, (3,3), 0, borderType=cv2.BORDER_CONSTANT)
    ret, ax = cv2.threshold(a,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ax= cv2.bitwise_not(ax)
    ax=pcv.fill_holes(ax)
    ax= cv2.erode(ax, kernal, iterations=4)
    

    label_a= measure.label(ax)

    props_a = measure.regionprops_table(label_a, 
                                  properties=['label','area','eccentricity'])
    lab_a = props_a['label']
    ecc_a = props_a['eccentricity']
    size_a = props_a['area']
    nb_comp_a = lab_a.max()
    min_size_a = 10
    e=0.8
    a_1 = np. zeros(( label_a.shape ))
    for i in range(0, nb_comp_a):
            if (size_a[i] >= min_size_a and ecc_a[i]<=e):
                a_1[label_a == i + 1] = 255
    
    
    label_a_1= measure.label(a_1)
    props_a_1 = measure.regionprops_table(label_a_1, 
                                  properties=['label','equivalent_diameter', 'centroid'])

    df_a_1= pd.DataFrame(props_a_1)


    a_2 = np.zeros(( label_a.shape ), dtype=np.double)

    for i in (df_a_1.index): 
    
        rr, cc = disk((df_a_1["centroid-0"][i],df_a_1["centroid-1"][i]), (df_a_1["equivalent_diameter"][i])/2, shape=a_2.shape)
        a_2[rr, cc] = 1


    my_list.append(a_2)
    
    label_a_2= measure.label(a_2)
    
    
   
    props_a_2= measure.regionprops_table(label_a_2, a,
                                     properties=['label', 'mean_intensity', 'centroid', 'equivalent_diameter'])
    df_a_2= pd.DataFrame(props_a_2) 
    
    df_a_2['equivalent_diameter']=df_a_2['equivalent_diameter']*pixcon
   
    
    if l==0:
        df_final["Droplet_Number"]=df_a_2["label"]
    
    df_final['Droplet equivalent_diameters of img no. '+str(l+1)] = float('nan')
    df_final['Droplet mean_intensity of img no. '+str(l+1)] = float('nan')
    df_final['Droplet volume of img no. '+str(l+1)] = float('nan')
    df_final['y_cor of img no. '+str(l+1)] = float('nan')
    df_final['x_cor of img no. '+str(l+1)] = float('nan')
    
    if l==0:
        for i in (df_a_2.index):
            df_final['Droplet equivalent_diameters of img no. '+str(l+1)][i] = df_a_2['equivalent_diameter'][i]
            df_final['Droplet mean_intensity of img no. '+str(l+1)][i] = 255 - df_a_2['mean_intensity'][i]
            df_final['Droplet volume of img no. '+str(l+1)][i] = 4/3*3.14* ((df_a_2['equivalent_diameter'][i]/2)**3)*((2-((1-(h/df_a_2['equivalent_diameter'][i]))**2)) * (2+(h/df_a_2['equivalent_diameter'][i])))
            df_final['y_cor of img no. '+str(l+1)] = df_a_2["centroid-0"]
            df_final['x_cor of img no. '+str(l+1)] = df_a_2["centroid-1"]
    
    if l>0:
       
            for k in (df_final.index):
                for i in (df_a_2.index):
                    if ((abs(df_final['y_cor of img no. '+str(l)][k] - df_a_2["centroid-0"][i]) <= c ) and (abs(df_final['x_cor of img no. '+str(l)][k] - df_a_2["centroid-1"][i]) <= c )):
                        df_final['Droplet equivalent_diameters of img no. '+str(l+1)][k]=df_a_2['equivalent_diameter'][i]
                        df_final['Droplet mean_intensity of img no. '+str(l+1)][k]= 255 - df_a_2['mean_intensity'][i]
                        df_final['Droplet volume of img no. '+str(l+1)][k]= 4/3*3.14* ((df_a_2['equivalent_diameter'][i]/2)**3)*((2-((1-(h/df_a_2['equivalent_diameter'][i]))**2)) * (2+(h/df_a_2['equivalent_diameter'][i])))
                        df_final['y_cor of img no. '+str(l+1)][k] = df_a_2["centroid-0"][i]
                        df_final['x_cor of img no. '+str(l+1)][k] = df_a_2["centroid-1"][i]
                        break
                    

    
    l=l+1

    
    del df_a_2
    del df_a_1
   



l=l-1

final_Dia["Diameter_Droplet_Number"]= df_final["Droplet_Number"]
final_Vol["Volume_Droplet_Number"]= df_final["Droplet_Number"]
final_Concentration["Concentration_Droplet_Number"]= df_final["Droplet_Number"]

# max= df_final['Droplet mean_intensity of img no. '+str(l+1)]
# min= df_final['Droplet mean_intensity of img no. '+str(2)]

max=255
min=0

for i in range(l):
    final_Dia['Image number '+ str(i+1)]= df_final['Droplet equivalent_diameters of img no. '+str(i+1)]
    final_Vol['Image number '+ str(i+1)]= df_final['Droplet volume of img no. '+str(i+1)]
    final_Concentration['Image number '+ str(i+1)]= (df_final['Droplet mean_intensity of img no. '+str(i+1)]-min)/(max-min)

final = final_Dia.append([final_Vol, final_Concentration])

second_column = final.pop('Volume_Droplet_Number')
final.insert(1, 'Volume_Droplet_Number', second_column)

third_column = final.pop("Concentration_Droplet_Number")
final.insert(2, 'Concentration_Droplet_Number', third_column)

final = final.T



final.to_csv(r"C:\Users\Nishanth\Desktop\drying droplet excel results\final.csv")
final_Dia.to_csv(r"C:\Users\Nishanth\Desktop\drying droplet excel results\df_final.csv")


# b = len(my_list)
# a=0
# for a in range(8):
#       cv2.imshow('Original Image'+str(a+1), my_list[a])
#       cv2.waitKey(0)
#       cv2.destroyAllWindows()

plt.imshow(my_list[202])
    
# cv2.imshow('Original Image'+str(173), my_list[0])
# cv2.imshow('Original Image'+str(174), my_list[170])
# cv2.imshow('Original Image'+str(175), my_list[171])
# cv2.imshow('Original Image'+str(176), my_list[172])
# cv2.imshow('Original Image'+str(177), my_list[173])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(df_final.index)
# b=df_final.index