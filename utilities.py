import os
import pandas as pd
class Utilities:
    @staticmethod
    def append_in_file(row):
        df = pd.DataFrame(row)
        df.to_csv('data.csv', mode='a', header=False, index=False)