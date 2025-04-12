import numpy as np

# in python , when we slice a list , a new list is created , so if we change that new list , that will not affect the original one
# but in case of numpy , if we slice a list , and assingn that sliced list into a variable , then if that sliced list is changed then the original one will 
#   also be changed 

# np.arange() , this method can take , one , two or three arguments ,,,,,,,in case we use three arguments , then the third argument indicates the steep 

#when we reshape a numpy array , the array itself does not gets modified , rather a copy of the modified (reshaped) array is given