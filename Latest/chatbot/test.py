
import pandas as pd
# from classification import perform_analysis

# from werkzeug.datastructures import FileStorage
def get_features():
    # Read the CSV file
    file_path = 'tempsy/f'
    df = pd.read_csv(file_path)
    print("")
    # Get column names, data types, and number of distinct values
    columns_info = {'name':[],'datatype':[],'distinct_values':[]}
    i=1
    for column in df.columns:
        columns_info['name'].append(column)
        columns_info['datatype'].append(str(df[column].dtype))
        columns_info['distinct_values'].append(df[column].nunique())
    return columns_info

# def get_best_model(type,target):
#     file=FileStorage(filename='f', stream=open('tempsy/f', 'rb'))
#     if type=='classification':
#         return(perform_analysis(file,target))

# bla=get_best_model('classification','Outcome')
# print(bla)
