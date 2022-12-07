
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import pandas as pd
from scipy.signal import savgol_filter
import random


kernal = np.ones((2,2),np.uint8)

#Declaration of variables
vo=[]
file_list=[]
filef_list=[]
s=[]
dia=[]
vol=[]
conc=[]
pol=[]
regn=[]
path=[]
pathf=[]

#Asking for inputs 

xy=int(input("Enter number of field of views: "))
for i in range(xy):
    loc= input("Enter the path of brightfield folder number "+ str(i+1)+" : ")
    loc1= input("Enter the path of polarized folder number "+ str(i+1)+" : ")
    path.append(loc[1:-1])
    pathf.append(loc1[1:-1])

#Saving the folder paths to file_list and filef_list

for j in range(xy):
    file_list.append(glob.glob( path[j]+"\*.*"))
    
    filef_list.append(glob.glob( pathf[j]+"\*.*"))

#Displaying the first image of the folders

im = cv2.imread(file_list[0][0])
cv2.imshow('test', im)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Scaling down the images to app sizes 
 
m=int(input("Enter scale of image in PERCENT that you want to work with: "))
scale_percent = m
width = int(im.shape[1] * scale_percent / 100)
height = int(im.shape[0] * scale_percent / 100)
dsize = (width, height)
im = cv2.resize(im, dsize)
# cv2.imshow('test', im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


lm=0

#Resizing all the images to the entered scale
for j in range(xy):
    
    im = cv2.imread(file_list[j][0])
    im = cv2.resize(im, dsize) 
    cv2.imshow('test', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Choosing the regions 

    reg=int(input("Enter number of regions you would like to choose in the current image"))
    regn.append(reg)

#Cropping out the droplets from all the droplets        
    for i in range(reg):
        print("choose region"+str(i+1))
        r = cv2.selectROI(im)
        s.append(r)
        imCrop = im[(s[i + lm][1]):(s[i + lm][1]+s[i + lm][3]), (s[i + lm][0]):(s[i + lm][0]+s[i + lm][2])]
        cv2.imshow("Image", imCrop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    lm= lm+reg

#entering the number of erode iterations on the image if you would like to keep 0 all the time
#uncomment the next line of code and comment the line after that  
#q=0
q=int(input("Enter the number of erode iterations you would like on the image : "))


#Running iterations through all the images in all the folders

lol=0
for fol in range(xy):
    for inimg in range(regn[fol]): 
        
                kol=len(file_list[fol])

#Declaring the dataframes for holding the data
                df_final= pd.DataFrame()
                final_Dia= pd.DataFrame()
                final_Vol= pd.DataFrame()
                final_Int= pd.DataFrame()
                final_Conc= pd.DataFrame()
                final_Pol_Int= pd.DataFrame()
                final= pd.DataFrame()  

#declaring a few constants 
                
                ret=0
                l=0
                b=100
                n=0
                pixcon=100/178
                h=25
                h1=50
                c=0.25
                eqDia=0
                # vo=0
             

#here the image cleaning starts, croppings->blurring->thresh holding->eroding              
                for file in range (kol):   
                    
                    #print(file)     
                    a= cv2.imread(file_list[fol][file],0)
                    
                    
                    a = cv2.resize(a, dsize)
                    
                    
                    a = a[int(s[lol][1]):int(s[lol][1]+s[lol][3]), int(s[lol][0]):int(s[lol][0]+s[lol][2])]
                        
#uncomment these lines and change image path if youd like to see each image after this iterations                         
                    # cv2.imshow('1', a)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    #cv2.imwrite(r"C:\Users\Nishanth\Desktop\New folder (3)\1 "+str(l+1)+".png", a)
                
                    
                    a=cv2.GaussianBlur(a, (3,3), 0, borderType=cv2.BORDER_CONSTANT)
                    
                    ret, ax = cv2.threshold(a,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    
                   
                    
                    ax= cv2.bitwise_not(ax)
#uncomment these lines and change image path if youd like to see each image after this iterations  
                    # cv2.imshow('1', ax)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    #cv2.imwrite(r"C:\Users\Nishanth\Desktop\New folder (3)\2 "+str(l+1)+".png", ax)
                 
                    
                    
                    ax= cv2.erode(ax, kernal, iterations=q)
#uncomment these lines and change image path if youd like to see each image after this iterations                      
                    # cv2.imshow('1', ax)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    #cv2.imwrite(r"C:\Users\Nishanth\Desktop\New folder (3)\3 "+str(l+1)+".png", ax)
                
#Here the image labelling and analysis occurs and all non droplets are eliminated              
                
                    label_a= measure.label(ax)
                
                    props_a = measure.regionprops_table(label_a, 
                                                  properties=['label','area','eccentricity'])
                    lab_a = props_a['label']
                    ecc_a = props_a['eccentricity']
                    size_a = props_a['area']
                    nb_comp_a = lab_a.max()
                    min_size_a = 880
                    e=0.5
                    a_1 = np. zeros(( label_a.shape ))
                    for i in range(0, nb_comp_a):
                            if (size_a[i] >= min_size_a and ecc_a[i]<=e):
                                a_1[label_a == i + 1] = 255
 
#uncomment these lines and change image path if youd like to see each image after this iterations                      
                    # cv2.imshow('1', a_1)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    #cv2.imwrite(r"C:\Users\Nishanth\Desktop\New folder (3)\4 "+str(l+1)+".png", a_1)
                    
                    label_a_1= measure.label(a_1)
                    props_a_1 = measure.regionprops_table(label_a_1, 
                                                  properties=['bbox' ,'centroid'])
                
                    df_a_1= pd.DataFrame(props_a_1)
                
                
                    a_2 = np.zeros(( label_a.shape ), dtype=np.double)
                
                    
                    g=int(df_a_1["centroid-0"])
                    f=int(df_a_1["centroid-1"])
                    d=int((df_a_1["bbox-2"]-df_a_1["bbox-0"])/2)
                    d1=int((df_a_1["bbox-3"]-df_a_1["bbox-1"])/2)
                    a_2 = cv2.ellipse(a_2,(f,g),(d1,d),0,0,360,(255,255,255),-1)
                    
                   
                    
                    label_a_2= measure.label(a_2)
                    
#here all the information is gathered from the images and sorted in dataframes               
                   
                    props_a_2= measure.regionprops_table(label_a_2, a,
                                                      properties=['label', 'mean_intensity', 'centroid', 'equivalent_diameter'])
                    df_a_2= pd.DataFrame(props_a_2) 
                    
                    df_a_2['equivalent_diameter']=df_a_2['equivalent_diameter'][0] * pixcon * 100 / m
                   
                    
                    if l==0:
                        df_final["Droplet_Number"]=df_a_2["label"]
                    
                    df_final['Droplet equivalent_diameters of img no. '+str(l+1)] = float('nan')
                    df_final['Droplet mean_intensity of img no. '+str(l+1)] = float('nan')
                    df_final['Droplet mean_polarized_intensity of img no. '+str(l+1)] = float('nan')
                    df_final['Droplet volume of img no. '+str(l+1)] = float('nan')
                    df_final['Droplet concentration of img no. '+str(l+1)] = float('nan')
                    df_final['y_cor of img no. '+str(l+1)] = float('nan')
                    df_final['x_cor of img no. '+str(l+1)] = float('nan')
                    
                    eqDia= df_a_2['equivalent_diameter'][0]
                    meanInt = df_a_2['mean_intensity'][0]
                    
                    
                    if l==0:
                        
                        df_final['Droplet equivalent_diameters of img no. '+str(l+1)] = eqDia
                        df_final['Droplet mean_intensity of img no. '+str(l+1)] = 255 - meanInt
                        
                        
                        df_final['Droplet volume of img no. '+str(1)] = 4/3 * 3.14 * ((eqDia/2)**3)* 3/2 *(h/eqDia)*(1 - (2 * (h/eqDia)) + 5/3 * (h/eqDia)**2 + 3.14/2 * (h/eqDia)*(1- (h/eqDia)))
                            
                        if (eqDia <= h):                            
                                df_final['Droplet volume of img no. '+str(l+1)] = 4/3*3.14*((eqDia/2)**3)
                                print("its triggered")
                            
                            
                        vo=df_final['Droplet volume of img no. '+str(1)][0]
                        print("new vo is "+ str(vo))
                        df_final['Droplet concentration of img no. '+str(1)]= (vo/df_final['Droplet volume of img no. '+str(1)][0])*c
                        
                    
                    if l>0:
                       
                            
                            df_final['Droplet equivalent_diameters of img no. '+str(l+1)]=eqDia
                            df_final['Droplet mean_intensity of img no. '+str(l+1)]= 255 - meanInt
                                                                           
                            df_final['Droplet volume of img no. '+str(l+1)] = 4/3 * 3.14 * ((eqDia/2)**3)* 3/2 *(h/eqDia)*(1 - (2 * (h/eqDia)) + 5/3 * ((h/eqDia)**2) + 3.14/2 * (h/eqDia)*(1- (h/eqDia)))

#to see if the the droplet dia is less than the height of channel
                            if (eqDia <= float(h)):                            
                                df_final['Droplet volume of img no. '+str(l+1)] = (4/3) * 3.14 * ((eqDia/2)**3)
                                print("its triggered",df_a_2['equivalent_diameter'][0],h1)
                                #print(((2-((1-(h/df_a_2['equivalent_diameter'][0]))**2)), (2+(h/df_a_2['equivalent_diameter'][0]))))
                                      
                            
                            df_final['Droplet concentration of img no. '+str(l+1)]= (vo/df_final['Droplet volume of img no. '+str(l+1)])*c
                                     
                            
                    #print(file)     
                    af= cv2.imread(filef_list[fol][file],0)
                       
                  
                    af = cv2.resize(af, dsize)
                    
                   
                    af = af[int(s[lol][1]):int(s[lol][1]+s[lol][3]), int(s[lol][0]):int(s[lol][0]+s[lol][2])]
                    
                    props_af = measure.regionprops_table(label_a_2, af, properties=['label', 'mean_intensity'])
                                                  
                
                    df_af= pd.DataFrame(props_af)
                    
                    df_final['Droplet mean_polarized_intensity of img no. '+str(l+1)]= df_af['mean_intensity']
                          
                
                    
                    l=l+1
                
                             
                
                
    
                
                l=l-1
                
                final_Dia["Diameter_Droplet_Number"]= df_final["Droplet_Number"]
                final_Vol["Volume_Droplet_Number"]= df_final["Droplet_Number"]
                final_Int["Intensity_Droplet_Number"]= df_final["Droplet_Number"]
                final_Conc["Concentration_Droplet_Number"]= df_final["Droplet_Number"]
                final_Pol_Int["Polarized_Intensity_Droplet_Number"]= df_final["Droplet_Number"]
                
              
                for i in range(l):
                    final_Dia['Image number '+ str(i+1)]= df_final['Droplet equivalent_diameters of img no. '+str(i+1)]
                    final_Vol['Image number '+ str(i+1)]= df_final['Droplet volume of img no. '+str(i+1)]
                    final_Int['Image number '+ str(i+1)]= df_final['Droplet mean_intensity of img no. '+str(i+1)]
                    final_Conc['Image number '+ str(i+1)]= df_final['Droplet concentration of img no. '+str(i+1)]
                    final_Pol_Int['Image number '+ str(i+1)]= df_final['Droplet mean_polarized_intensity of img no. '+str(i+1)]

#values are sorted into a final dataframe                
                final = final_Dia.append([final_Vol, final_Int,final_Conc,final_Pol_Int])
                
                second_column = final.pop('Volume_Droplet_Number')
                final.insert(1, 'Volume_Droplet_Number', second_column)
                
                third_column = final.pop("Intensity_Droplet_Number")
                final.insert(2, 'Intensity_Droplet_Number', third_column)
                
                fourth_column = final.pop("Concentration_Droplet_Number")
                final.insert(3,"Concentration_Droplet_Number" , fourth_column)
                
                fifth_column = final.pop("Polarized_Intensity_Droplet_Number")
                final.insert(4,"Polarized_Intensity_Droplet_Number" , fifth_column)
                
                
                final = final.reset_index(drop=True)
                
                final = final.T
                

#converting the dataframe into a CSV folder and placing it into a file location
                
                final.to_csv(r"C:\Users\Nishanth\Desktop\drying droplet excel results\final"+str(lol)+".csv")
                
               
                dia.append(final[0][5:].values)
                vol.append(final[1][5:].values)
                conc.append(final[3][5:].values)
                pol.append(final[4][5:].values)
                
                
                lol=lol+1

#Cleaning up the dataframes                
                del a
                del ax
                del a_1
                del a_2
                del af
                del label_a
                del label_a_1
                del label_a_2              
                del df_af
                del df_a_2
                del df_a_1
                del df_final
                del final_Dia
                del final_Vol
                del final_Int
                del final_Conc
                del final_Pol_Int
                del final
               
#Declaring the graphing function variables           
#Graphing section
ysmooth=[]
xsmooth=[]
x1smooth=[]
x2smooth=[]
diasmooth=[]
polgrad=[]
polgrad2=[]
polgradsmooth=[]
avgconc=[]
stddevconc=[]
avgpoint=[]
points=[]
pointsconc=[]
A=0
B=0
C=0
D=0
E=0

#iterating through each droplet
for i in range (lm):

#defining colours for each droplet                
    col1 = random.random()
    col2 = random.random()
    col3 = random.random()
    Color = np.array([col1, col2, col3])

#Plotting volume                   
    plt.subplot(2, 3, (1))
    plt.tick_params(axis='x', colors='white') 
    xsmooth.append(savgol_filter(vol[i], 11, 0))  
    plt.plot(xsmooth[i], c= Color)
    plt.xlabel('Image number')     
    plt.ylabel('Volume')
    plt.title('Volume') 
    
#plotting Polarized data
    plt.subplot(2, 3, (3))
    plt.tick_params(axis='x', colors='white')   
    x2smooth.append(savgol_filter(pol[i], 11, 0)) 
    polgrad.append(np.gradient(x2smooth[i]))
    polgrad2.append(np.gradient(polgrad[i]))
    plt.plot(x2smooth[i], c= Color)
    plt.xlabel('Image number')     
    plt.ylabel('ploarized data')
    plt.title('polarized data') 
                    
#Plotting polarized gradient
    
    plt.subplot(2, 3, (4))
    plt.tick_params(axis='x', colors='white')      
    polgradsmooth.append(savgol_filter(polgrad[i], 11, 0)) 
    plt.plot(polgradsmooth[i], c= Color)
    plt.xlabel('Image number')     
    plt.ylabel('ploarized grad')
    plt.title('polarized grad')  

#Plotting Diameter    
    plt.subplot(2, 3, (6))
    plt.tick_params(axis='x', colors='white')
    diasmooth.append(savgol_filter(dia[i], 3, 0))  
    plt.plot(diasmooth[i],c= Color)
    plt.xlabel('Image number')     
    plt.ylabel('Dia in Micron')
    plt.title('Diameter')  
    
    
#finding the values of A, B, C, D, E
    
    acheck=0
    bcheck=0
    ccheck=0
    dcheck=0
    echeck=0
    
    for j in range (l):
        if (acheck == 0):
            if (pol[i][j] > 1): # find values from pol IJ>0
                A=j
                acheck=1
        if (acheck==1 and bcheck == 0):
            if (x2smooth[i][j] >= x2smooth[i][j-1] and x2smooth[i][j] >= x2smooth[i][j+1]): #max of the x2smooth curve
                B=j
                bcheck=1
        if (bcheck==1 and ccheck == 0):
            if (polgradsmooth[i][j] <= polgradsmooth[i][j-1] and polgradsmooth[i][j] <= polgradsmooth[i][j+1]):
                C=j
                ccheck=1
        if (ccheck==1 and dcheck == 0):
            if (polgradsmooth[i][j] >= polgradsmooth[i][j-1] and polgradsmooth[i][j] >= polgradsmooth[i][j+1]):
                D=j
                dcheck=1
                
        if (dcheck==1 and echeck == 0):
            if (polgradsmooth[i][j] <= polgradsmooth[i][j-1] and polgradsmooth[i][j] <= polgradsmooth[i][j+1]):
                E=j
                echeck=1

#Entering the values of A B C D E and corresponding conc in points and pointsconc                
    points.append([A,B,C,D,E])
    pointsconc.append([conc[i][A], conc[i][B], conc[i][C], conc[i][D], conc[i][E]])


#graphing Concentration of A B C D E vs image number of A B C D E
    plt.subplot(2, 3, (5))
    for j in range (5):
        plt.scatter(j , pointsconc[i][j], c= Color)
    plt.xlabel('Concentration points')     
    plt.ylabel('Concentration')
    plt.title('Image points') 
    
#Graphing concentration curves along with the A B C D E positions as well    
    plt.subplot(2, 3, (2))
    plt.tick_params(axis='x', colors='white') 
    x1smooth.append(savgol_filter(conc[i], 11, 0))  
    plt.plot(x1smooth[i], c= Color)
    for j in range (5):
          plt.scatter(points[i][j], pointsconc[i][j], c= Color)
    plt.xlabel('Image number')     
    plt.ylabel('Conc')
    plt.title('Conc')

plt.show()    

#Saving the AVG and the STD.DEV of the A B C D E conc points into avgconc and stddevconc 
avgconc= np.mean(pointsconc, axis=0)
stddevconc= np.std(pointsconc, axis=0)



 



