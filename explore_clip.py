import pandas as pd
import IPython


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



#### Filter df by column? to get those where data is good.

### 


#filtered_df = 


#### form a small_df by filtering the filtered_df by clip_predictor_columns + [output_column]

#small_df = 




IPython.embed()
