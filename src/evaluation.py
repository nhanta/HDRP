import os
import pandas as pd
import re
import argparse

parser = argparse.ArgumentParser(
                    prog='Training models using Machine Learning',
                    description='Logistic Regression, Decision Tree, and SVM',
                    epilog='Help')

parser.add_argument('input', metavar='i', type=str, help='Input path')
parser.add_argument('output', metavar='o', type=str, help='Output path')
args = parser.parse_args()
# Input and output folders
input_folder = args.input
output_excel = args.output + '/ALL_REPORT.csv'


headerList = ['Model', 'Sensitivity', 'Specificity', 'Accuracy', 'AUC', 'running_time', 'n_feature']
save_frame = pd.DataFrame(columns=headerList)
list_dir = os.listdir(input_folder)

# Loop through all files in the input folder
for i in range(0, len(list_dir)):
    # print info working
    current_item = i + 1
    print("Processing item", current_item, "of", len(list_dir))
    filename = list_dir[i]

    # regex get file name remove _evaluation.csv
    match = re.search(r'([\w_]+)_evaluations\.csv', filename)
    if match:
        print("match: ",filename)
        # if filename.endswith('_evaluations.csv'): # find using regex or using endswith
        model_name = match.group(1)  # get model name
        input_csv = os.path.join(input_folder, filename)
        data = pd.read_csv(input_csv)
        rows_data = data.iloc[0].astype(float).round(2).astype(str)  + " " + data.iloc[1].astype(str)
        sensitivity = rows_data[1]
        specificity = rows_data[2]
        accuracy = rows_data[3]
        auc = rows_data[4]
        running_time = ""
        n_feature = ""
        new_row = {'Model': model_name, 'Sensitivity': sensitivity, 'Specificity': specificity, 'Accuracy': accuracy,
                   'AUC': auc, 'running_time': running_time, 'n_feature': n_feature}
        save_frame.loc[len(save_frame)] = new_row

save_frame.to_csv(output_excel)
print("done")