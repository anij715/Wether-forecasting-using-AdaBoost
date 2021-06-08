import pandas as pd # for data manipulation
import numpy as np # for data manipulation

from sklearn.model_selection import train_test_split # for splitting the data into train and test samples
from sklearn.metrics import classification_report # for model evaluation metrics
from sklearn.ensemble import AdaBoostClassifier # for AdaBoost model
from sklearn.tree import DecisionTreeClassifier

import plotly.express as px  # for data visualization
import plotly.graph_objects as go # for data visualization

# Set Pandas options to display more columns
pd.options.display.max_columns=50

# Read in the weather data csv
df=pd.read_csv('C:\Program Files\Python39\DATA\Projects\Ada-boost\Aus-weather\weatherAUS.csv', encoding='utf-8')

# Drop records where target RainTomorrow=NaN
df=df[pd.isnull(df['RainTomorrow'])==False]

# For other columns with missing values, fill them in with column mean
df=df.fillna(df.mean())

# Create a flag for RainToday and RainTomorrow, note RainTomorrowFlag will be our target variable
df['RainTodayFlag']=df['RainToday'].apply(lambda x: 1 if x=='Yes' else 0)
df['RainTomorrowFlag']=df['RainTomorrow'].apply(lambda x: 1 if x=='Yes' else 0)

# Show a snaphsot of data
#print(df)

##### Step 1 - Select data for modeling
X=df[['WindGustSpeed', 'Humidity3pm']]
y=df['RainTomorrowFlag'].values


##### Step 2 - Create training and testing samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


##### Step 3
# Set model and its parameters
model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_leaf=1000, max_depth=1),
                           n_estimators=50, # default=50
                           learning_rate=1.0, # default=1. 
                           algorithm='SAMME', # SAMME' - discreate, 'SAMME.R' - real
                           random_state=0, # random state for reproducibility
                          )

# Fit the model
clf = model.fit(X_train, y_train)


##### Step 4
# Predict class labels on training data
pred_labels_tr = model.predict(X_train)
# Predict class labels on a test data
pred_labels_te = model.predict(X_test)


##### Step 5 - Model summary
# Basic info about the model
print('*************** Tree Summary ***************')
print('No. of classes: ', clf.n_classes_)
print('Classes: ', clf.classes_)
print('No. of Estimators: ', len(clf.estimators_))
print('Base Estimator: ', clf.base_estimator_)
print('--------------------------------------------------------')
print("")

print('*************** Evaluation on Test Data ***************')
score_te = model.score(X_test, y_test)
print('Accuracy Score: ', score_te)
# Look at classification report to evaluate the model
print(classification_report(y_test, pred_labels_te))
print('--------------------------------------------------------')
print("")

print('*************** Evaluation on Training Data ***************')
score_tr = model.score(X_train, y_train)
print('Accuracy Score: ', score_tr)
# Look at classification report to evaluate the model
print(classification_report(y_train, pred_labels_tr))
print('--------------------------------------------------------')

def Plot_3D(X, X_test, y_test, clf, x1, x2, mesh_size, margin):
            
    # Specify a size of the mesh to be used
    mesh_size=mesh_size
    margin=margin

    # Create a mesh grid on which we will run our model
    x_min, x_max = X.iloc[:, 0].fillna(X.mean()).min() - margin, X.iloc[:, 0].fillna(X.mean()).max() + margin
    y_min, y_max = X.iloc[:, 1].fillna(X.mean()).min() - margin, X.iloc[:, 1].fillna(X.mean()).max() + margin
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)
            
    # Calculate predictions on grid
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    # Create a 3D scatter plot with predictions
    #fig = px.scatter_3d(x=X_test[x1], y=X_test[x2], z=y_test,
    fig = px.scatter_3d(x=[], y=[], z=[],
                     opacity=0.8, color_discrete_sequence=['black'])

    # Set figure title and colors
    fig.update_layout(#title_text="Scatter 3D Plot with Adaboost Prediction Surface",
                      paper_bgcolor = 'white',
                      scene = dict(xaxis=dict(title=x1,
                                              backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0'),
                                   yaxis=dict(title=x2,
                                              backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0'
                                              ),
                                   zaxis=dict(title='Probability of Rain Tomorrow',
                                              backgroundcolor='lightgrey',
                                              color='black', 
                                              gridcolor='#f0f0f0',
                                              range=[0, 1],
                                              tickmode = 'linear',
                                              tick0 = 0,
                                              dtick = 0.2
                                              )))
    
    # Update marker size
    fig.update_traces(marker=dict(size=1))

    # Add prediction plane
    fig.add_traces(go.Surface(x=xrange, y=yrange, z=Z, name='AdaBoost Prediction',
                              colorscale='Bluered',
                              reversescale=True,
                              showscale=False, 
                              contours = {"z": {"show": True, "start": 0.5, "end": 0.9, 
                                                "size": 0.5, "color":"white"}}))
    fig.show()
    return fig
  
# Call the above fucntion to generate a graph
fig = Plot_3D(X, X_test, y_test, clf, x1='WindGustSpeed', x2='Humidity3pm', mesh_size=1, margin=1)