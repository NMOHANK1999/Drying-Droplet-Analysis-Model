# Drying-Droplet-Analysis-Model
Model to study the phase change of drying liquid droplets in a Microfluidic setup. 
Evaluated changes to phase by noticing changes/rate of changes in physical features of the time series droplet images.
The main "physical feature" in question is the polarized intensity of the droplets, which give us insight into its current phase.
You can see that there are multiple codes in this repository. Each code does the same task but the difference lies in the methodology of
deriving this information, some codes analyse droplets one at a time while other codes analyse all the droplets in the image simultaneously.
Libraries Used: Cv2, glob, numpy, matplotlib, skimage, pandas, scipy.signal, random
