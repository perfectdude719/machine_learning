import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt



#initialize variables to hold data#
class_g=[]
class_h=[]

#read the data#
with open('data.txt','r') as file:   #alternatively we can use np.genfromtxt()...prefered to do it manually for practice 
    for line in file:
        values=line.strip().split(',')
        data = list(map(float, values[:-1])) #converting to integers for later processing 
        if values[-1]=='g':
            class_g.append(data + [1]) #class gamma label
        else:
            class_h.append(data + [0]) #class hedron label
            
#find the imbalance #
len_g=len(class_g)
len_h=len(class_h)
diff=len_g-len_h
print(f"the length of class g is {len_g}")
print(f"the length of class h is {len_h}")
print(f"we need to set aside {diff} random examples from class g to balance the data")

#fixing the imbalance#
np. random.seed(70) # generate a seed for reproducibility of results (hn5tar random) 
index_of_putaside = np.random.choice(len_g, diff, replace=False)  # Select 'diff' unique random indices
print(index_of_putaside[0:4])
downsized_g=[item for i,item in enumerate(class_g) if i not in index_of_putaside] #downsized to elements not in the putaside array
print(len(downsized_g))

#convert to numpy array for proper utilization
gamma=np.array(downsized_g)
hedron=np.array(class_h)
full_data=np.concatenate((gamma,hedron),axis=0)

#visiuallize the data#
print(f"the first three isntances of each matrix gamma features:\n{gamma[:3]}\n hedron:\n{hedron[:3]}")

#split the data into  70% training, 15% test, 15% training      #alternatively np.split()
np.random.shuffle(full_data)

training_set_index=int(0.7*len(full_data))
test_set_index=int(0.15*len(full_data)) +training_set_index

training_set=full_data[0:training_set_index]
test_set=full_data[training_set_index:test_set_index]
validation_set=full_data[test_set_index:]

# Prepare features and labels for training and testing
X_train = training_set[:, :-1]  # Features (all columns except the last)
y_train = training_set[:, -1]   # Labels (last column)

X_test = test_set[:, :-1]  # Features for test set
y_test = test_set[:, -1]   # Labels for test set

X_validation = validation_set[:, :-1]  # Features (all columns except the last)
y_validation = validation_set[:, -1]   # Labels (last column)

# Apply KNN
f1=[]
for k in range(1,101):
    knn = KNeighborsClassifier(n_neighbors=k)  # You can adjust the number of neighbors
    knn.fit(X_train, y_train)  # Train the model
    y_pred_val = knn.predict(X_validation)
    
    # Calculate additional metrics for the validation set
    accuracy = accuracy_score(y_validation, y_pred_val)
    precision = precision_score(y_validation, y_pred_val)
    recall = recall_score(y_validation, y_pred_val)
    cm=confusion_matrix(y_validation,y_pred_val)
    f1.append(f1_score(y_validation, y_pred_val))

best_k_value=f1.index(max(f1))+1
print(f"the best k index is {best_k_value}")

# Visualization of F1 scores
plt.plot(range(1,101), f1, marker='o')
plt.title('F1 Score vs. K Value')
plt.xlabel('K Value')
plt.ylabel('F1 Score')
plt.grid()
plt.show()