# -*- coding: utf-8 -*-
"""
Code with all options
"""

import cv2
import glob

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import pandas as pd
from plantcv import plantcv as pcv
from skimage.draw import disk
 

df_final= pd.DataFrame()
final_Dia= pd.DataFrame()
final_Vol= pd.DataFrame()
final_Int= pd.DataFrame()
final_Conc= pd.DataFrame()
final_Pol_Int= pd.DataFrame()

final= pd.DataFrame()

kernal = np.ones((2,2),np.uint8)
ret=0
l=0
my_list=[]
my_listf=[]
vo=[]
x=[]
pixcon=100/178
h=25
c=0.25
b=100
n=0

im = cv2.imread(r"C:\Users\Nishanth\Desktop\test\drying_ssy_25mol_kg_8bit_c1xy7t001.tif")
cv2.imshow('test', im)
cv2.waitKey(0)
cv2.destroyAllWindows()


o=int(input("Enter 1 if you would like to change the scale of image: "))
if (o == 1):
    m=int(input("Enter scale of image in PERCENT that you want to work with: "))
    scale_percent = m
    width = int(im.shape[1] * scale_percent / 100)
    height = int(im.shape[0] * scale_percent / 100)
    dsize = (width, height)
    im = cv2.resize(im, dsize)
    cv2.imshow('test', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# o=int(input("Enter 1 if you would like to change the scale of image again: "))
# if (o==1):
#     m=input("Enter scale of image in PERCENT that you want to work with: ")
#     scale_percent = m
#     width = int(im.shape[1] * scale_percent / 100)
#     height = int(im.shape[0] * scale_percent / 100)
#     dsize = (width, height)
#     im = cv2.resize(im, dsize)
#     cv2.imshow('test', im)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    

p=int(input("Enter 1 if you would like to choose a specific area of the droplet to work with : "))
if (p==1):
    r = cv2.selectROI(im)
    imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    cv2.imshow("Image", imCrop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

q=int(input("Enter the number of erode iterations you would like on the image : "))


file_list = glob.glob(r"C:\Users\Nishanth\Desktop\test\*.*") 
print(file_list)

path = r"C:\Users\Nishanth\Desktop\test\*.*"
for file in glob.glob(path):   
    print(file)     
    a= cv2.imread(file,0)
    
    if (o == 1):
        a = cv2.resize(a, dsize)
    
    if (p==1):
        a = a[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        
        
    # cv2.imshow('1', a)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(r"C:\Users\Nishanth\Desktop\New folder (3)\1 "+str(l+1)+".png", a)

    
    a=cv2.GaussianBlur(a, (3,3), 0, borderType=cv2.BORDER_CONSTANT)
    
    ret, ax = cv2.threshold(a,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # thres=np.percentile(a.flat, 5)
    # ret, ax = cv2.threshold(a,thres,255,cv2.THRESH_BINARY)
    
    ax= cv2.bitwise_not(ax)
    # cv2.imshow('1', ax)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(r"C:\Users\Nishanth\Desktop\New folder (3)\2 "+str(l+1)+".png", ax)
 
    
    
    ax= cv2.erode(ax, kernal, iterations=q)
    
    # cv2.imshow('1', ax)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(r"C:\Users\Nishanth\Desktop\New folder (3)\3 "+str(l+1)+".png", ax)

    label_a= measure.label(ax)

    props_a = measure.regionprops_table(label_a, 
                                  properties=['label','area','eccentricity'])
    lab_a = props_a['label']
    ecc_a = props_a['eccentricity']
    size_a = props_a['area']
    nb_comp_a = lab_a.max()
    min_size_a = 900
    max_size_a =15000
    e=0.35
    a_1 = np. zeros(( label_a.shape ))
    for i in range(0, nb_comp_a):
            if (size_a[i] >= min_size_a and ecc_a[i]<=e and size_a[i] <= max_size_a):
                a_1[label_a == i + 1] = 255
    
    # cv2.imshow('1', a_1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #a_1=pcv.fill_holes(a_1)
    cv2.imwrite(r"C:\Users\Nishanth\Desktop\New folder (3)\4 "+str(l+1)+".png", a_1)
    
    label_a_1= measure.label(a_1)
    props_a_1 = measure.regionprops_table(label_a_1, 
                                  properties=['label','equivalent_diameter','bbox' ,'centroid'])

    df_a_1= pd.DataFrame(props_a_1)


    a_2 = np.zeros(( label_a.shape ), dtype=np.double)

    for i in (df_a_1.index):
        g=int(df_a_1["centroid-0"][i])
        f=int(df_a_1["centroid-1"][i])
        d=int((df_a_1["bbox-2"][i]-df_a_1["bbox-0"][i])/2)
        d1=int((df_a_1["bbox-3"][i]-df_a_1["bbox-1"][i])/2)
        a_2 = cv2.ellipse(a_2,(f,g),(d1,d),0,0,360,(255,255,255),-1)
    
        # rr, cc = disk((df_a_1["centroid-0"][i],df_a_1["centroid-1"][i]), (df_a_1["equivalent_diameter"][i])/2, shape=a_2.shape)
        # a_2[rr, cc] = 1
      
    cv2.imwrite(r"C:\Users\Nishanth\Desktop\New folder (3)\5 "+str(l+1)+".png", a_2)    


    my_list.append(a_2)
    
    label_a_2= measure.label(a_2)
    
    
   
    props_a_2= measure.regionprops_table(label_a_2, a,
                                     properties=['label', 'mean_intensity', 'centroid', 'equivalent_diameter'])
    df_a_2= pd.DataFrame(props_a_2) 
    
    df_a_2['equivalent_diameter']=df_a_2['equivalent_diameter'] * pixcon * 100 / m
   
    
    if l==0:
        df_final["Droplet_Number"]=df_a_2["label"]
    
    df_final['Droplet equivalent_diameters of img no. '+str(l+1)] = float('nan')
    df_final['Droplet mean_intensity of img no. '+str(l+1)] = float('nan')
    
    df_final['Droplet mean_polarized_intensity of img no. '+str(l+1)] = float('nan')
    
    df_final['Droplet volume of img no. '+str(l+1)] = float('nan')
    
    df_final['Droplet concentration of img no. '+str(l+1)] = float('nan')
    
    df_final['y_cor of img no. '+str(l+1)] = float('nan')
    df_final['x_cor of img no. '+str(l+1)] = float('nan')
    
    if l==0:
        for i in (df_a_2.index):
            df_final['Droplet equivalent_diameters of img no. '+str(l+1)][i] = df_a_2['equivalent_diameter'][i]
            df_final['Droplet mean_intensity of img no. '+str(l+1)][i] = 255 - df_a_2['mean_intensity'][i]
            df_final['Droplet volume of img no. '+str(l+1)][i] = 4/3*3.14* ((df_a_2['equivalent_diameter'][i]/2)**3)*((2-((1-(h/df_a_2['equivalent_diameter'][i]))**2)) * (2+(h/df_a_2['equivalent_diameter'][i])))
            vo.append(df_final['Droplet volume of img no. '+str(1)][i])
            
            df_final['Droplet concentration of img no. '+str(l+1)][i]= (vo[i]/ df_final['Droplet volume of img no. '+str(l+1)][i]) * c
            df_final['y_cor of img no. '+str(l+1)] = df_a_2["centroid-0"]
            df_final['x_cor of img no. '+str(l+1)] = df_a_2["centroid-1"]
    
    if l>0:
       
            for k in (df_final.index):
                for i in (df_a_2.index):
                    if ((abs(df_final['y_cor of img no. '+str(l)][k] - df_a_2["centroid-0"][i]) <= b ) and (abs(df_final['x_cor of img no. '+str(l)][k] - df_a_2["centroid-1"][i]) <= b )):
                        df_final['Droplet equivalent_diameters of img no. '+str(l+1)][k]=df_a_2['equivalent_diameter'][i]
                        df_final['Droplet mean_intensity of img no. '+str(l+1)][k]= 255 - df_a_2['mean_intensity'][i]
                        df_final['Droplet volume of img no. '+str(l+1)][k]= 4/3*3.14* ((df_a_2['equivalent_diameter'][i]/2)**3)*((2-((1-(h/df_a_2['equivalent_diameter'][i]))**2)) * (2+(h/df_a_2['equivalent_diameter'][i])))
                        
                        df_final['y_cor of img no. '+str(l+1)][k] = df_a_2["centroid-0"][i]
                        df_final['x_cor of img no. '+str(l+1)][k] = df_a_2["centroid-1"][i]
                        df_final['Droplet concentration of img no. '+str(l+1)][k]= (vo[k]/ df_final['Droplet volume of img no. '+str(l+1)][k])  * c
                        break
                    

    
    l=l+1

    
    del df_a_2
    del df_a_1

filef_list = glob.glob(r"C:\Users\Nishanth\Desktop\testf\*.*") 
print(filef_list)

pathf = r"C:\Users\Nishanth\Desktop\testf\*.*"
for filef in glob.glob(pathf):   
    print(filef)     
    af= cv2.imread(filef,0)
       
    if (o == 1): 
        af = cv2.resize(af, dsize)
    
    if (p==1):
        af = af[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    my_listf.append(af)
    
    label_af= measure.label(my_list[n])
    props_af = measure.regionprops_table(label_af, af, properties=['label','equivalent_diameter', 'centroid', 'mean_intensity'])
                                  

    df_af= pd.DataFrame(props_af)
    
    
    for k in (df_final.index):
        for i in (df_af.index):
            if ((abs(df_final['y_cor of img no. '+str(n+1)][k] - df_af["centroid-0"][i]) <= b ) and (abs(df_final['x_cor of img no. '+str(n+1)][k] - df_af["centroid-1"][i]) <= b )):
                
                df_final['Droplet mean_polarized_intensity of img no. '+str(n+1)][k]= df_af['mean_intensity'][i]
               
                
                break




    n=n+1
    del df_af





l=l-1

final_Dia["Diameter_Droplet_Number"]= df_final["Droplet_Number"]
final_Vol["Volume_Droplet_Number"]= df_final["Droplet_Number"]
final_Int["Intensity_Droplet_Number"]= df_final["Droplet_Number"]
final_Conc["Concentration_Droplet_Number"]= df_final["Droplet_Number"]
final_Pol_Int["Polarized_Intensity_Droplet_Number"]= df_final["Droplet_Number"]


max=255
min=0

for i in range(l):
    final_Dia['Image number '+ str(i+1)]= df_final['Droplet equivalent_diameters of img no. '+str(i+1)]
    final_Vol['Image number '+ str(i+1)]= df_final['Droplet volume of img no. '+str(i+1)]
    final_Int['Image number '+ str(i+1)]= df_final['Droplet mean_intensity of img no. '+str(i+1)]
    final_Conc['Image number '+ str(i+1)]= df_final['Droplet concentration of img no. '+str(i+1)]
    final_Pol_Int['Image number '+ str(i+1)]= df_final['Droplet mean_polarized_intensity of img no. '+str(i+1)]

final = final_Dia.append([final_Vol, final_Int,final_Conc,final_Pol_Int])

second_column = final.pop('Volume_Droplet_Number')
final.insert(1, 'Volume_Droplet_Number', second_column)

third_column = final.pop("Intensity_Droplet_Number")
final.insert(2, 'Intensity_Droplet_Number', third_column)

fourth_column = final.pop("Concentration_Droplet_Number")
final.insert(3,"Concentration_Droplet_Number" , fourth_column)

fifth_column = final.pop("Polarized_Intensity_Droplet_Number")
final.insert(4,"Polarized_Intensity_Droplet_Number" , fifth_column)

final = final.T



final.to_csv(r"C:\Users\Nishanth\Desktop\drying droplet excel results\final.csv")
# final_Dia.to_csv(r"C:\Users\Nishanth\Desktop\drying droplet excel results\df_final.csv")


# b = len(my_list)
# a=0
# for a in range(6):
#     cv2.imshow('Original Image'+str(a+1), my_list[a])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#plt.imshow(my_list[16])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# plt.imshow(my_listf[2])

# cv2.imshow('Original Image'+str(173), my_list[0])
# cv2.imshow('Original Image'+str(174), my_list[170])
# cv2.imshow('Original Image'+str(175), my_list[171])
# cv2.imshow('Original Image'+str(176), my_list[172])
# cv2.imshow('Original Image'+str(177), my_list[173])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(df_final.index)
# b=df_final.index