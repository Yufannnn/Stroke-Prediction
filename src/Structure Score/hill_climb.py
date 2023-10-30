import pandas as pd
from pgmpy.estimators import HillClimbSearch, BicScore

# Load your data
data = pd.read_csv('../../data/transformed_data.csv')

# Use Hill Climb Search
hc = HillClimbSearch(data)
best_model = hc.estimate(scoring_method=BicScore(data))

print(best_model.edges())
