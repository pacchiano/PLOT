import pandas as pd
import IPython
import numpy as np
import pickle
from datasets import DataSet


df = pd.read_csv("./main.CLIPZeroShotWithFeatures.txt", sep = "\t", engine="python")


informative_columns= ["Timestamp (ticks)", "Object Label", "Predicted Label", "Prediction Score", "Prediction Cardinality",
	"Prediction Error", "Object Mode"]

twod_object_columns = ["Rectangle Top", "Rectangle Bottom", "Rectable Left", "Rectangle Right",
	"Rectangle Width", "Rectangle Height"]

clip_predictor_columns = ["3D Height (m)", "3D Volume", "Confirmed Volume", "Intersection Volume",  "IOU",
	"Projection in View (%)", "Expanded Projection in View (%)", "Objects Occlusion (%)", "Hands Occlusion (%)",
	"Total Occlusion (%)", "Linear Head Speed (m/s)", "Angular Head Speed (rad/s)", "Distance to Object",
	"Viewing Angle", "Selectors.FullViews", "Selectors.FullViewsWithHeadVelocityAndOcclusion",
	"Selectors.ConfirmedFullViewsWithHeadVelocityAndOcclusion", "2D Projection IOU", "2D Expanded Projection IOU"]



#### Filtering and data-preprocessing

Xdata = df[clip_predictor_columns][50000:].values
### change percentages
Xdata[:,5]*= .01
Xdata[:,6]*= .01
Xdata[:,7]*= .01
Xdata[:,8]*= .01
Xdata[:,9]*= .01

### viewing angle
Xdata[:,13]*= 1.0/360

Xdata = Xdata/np.max(np.abs(Xdata), 0)

Xdata = Xdata.astype(float)

train_X = pd.DataFrame(Xdata[0:100000,:])
test_X = pd.DataFrame(Xdata[100000:, :])


Ydata = df["Prediction Score"][50000:].values

#filtered_df = 
train_y = pd.DataFrame(Ydata[0:100000])
test_y = pd.DataFrame(Ydata[100000:])

test_y = test_y.astype(float)


train_dataset = DataSet(train_X, train_y)
test_dataset = DataSet(test_X, test_y)


pickle.dump((train_dataset, test_dataset), open("./clip_datasets.p", "wb"))










IPython.embed()
