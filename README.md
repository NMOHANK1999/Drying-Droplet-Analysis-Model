# Drying-Droplet-Analysis-Model
Model to study the phase change of drying liquid droplets in a Microfluidic setup. 
Evaluated changes to phase by noticing changes/rate of changes in physical features of the time series droplet images.
The main "physical feature" in question is the polarized intensity of the droplets, which give us insight into its current phase.
You can see that there are multiple codes in this repository. Each code does the same task but the difference lies in the methodology of
deriving this information, some codes analyse droplets one at a time while other codes analyse all the droplets in the image simultaneously.
Libraries Used: Cv2, glob, numpy, matplotlib, skimage, pandas, scipy.signal, random
Inputs in the form of images as below:
![MMStack_Pos0 ome-10000](https://user-images.githubusercontent.com/109509567/206272637-2f3e8281-bfc5-4fb8-939f-f07cacc03e84.png)
![MMStack_Pos0 ome-10077](https://user-images.githubusercontent.com/109509567/206272642-b9c21ed5-bea4-4f59-9165-dbf89431dcc4.png)
![MMStack_Pos0 ome-10152](https://user-images.githubusercontent.com/109509567/206272643-9373ccaf-957c-4b18-912f-b0c6b5e4c785.png)

Results are in the form of Excel sheets and graphs an example is displayed below.
![graph results](https://user-images.githubusercontent.com/109509567/206271832-c699864d-df06-400a-8213-9df96b82f8fa.PNG)
