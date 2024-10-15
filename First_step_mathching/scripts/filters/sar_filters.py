# Libs
import pandas as pd


class SARFilters:
    @staticmethod
    def filter_sar_probabilities(df: pd.DataFrame) -> pd.DataFrame:
        df['first_probability'] = df['probabilities'].apply(lambda x: abs(x[0]))
        # Define the conditions as a function using only the first value of the probabilities list
        def filter_sar_data(row):
            dist = row['distance_to_shoreline'] #km
            first_probability = abs(row['probabilities'][0])  # Use the first value in the probabilities list
            
            # Apply the conditions 
            if dist <= 5:
                return first_probability >= 0.80
            elif 5 < dist <= 20:
                return first_probability >= 0.60
            elif dist > 20:
                return first_probability >= 0.25
            else:
                return False  # This condition should never be met

        # Apply the function to the DataFrame
        return df[df.apply(filter_sar_data, axis=1)] 

