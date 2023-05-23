


import os
import re
import pandas as pd

def format_confusion(path, confusion):

    data_rows = []
    removal_list = ["[", "]", "\n"]
    this_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    confusion_file = os.path.join(this_dir, path, confusion)
    with open(confusion_file, 'r') as f:
        lines = f.readlines()
    for l in lines:
        for s in removal_list:
            l = l.replace(s,'')     
           
            
        lc = re.sub("\s+", ",", l.strip())
        data_rows.append(lc.split(","))
    excel_out = os.path.join(this_dir, path,confusion[:-3]  + ".xlsx")

    df = pd.DataFrame(data_rows)
    df.to_excel(excel_out)
        
def format_confusion(path, classification):

    data_rows = []
    removal_list = ["\n"]
    this_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    classification_file = os.path.join(this_dir, path, classification)
    with open(classification_file, 'r') as f:
        lines = f.readlines()
    for l in lines[2:]:
        for s in removal_list:
            l = l.replace(s,'')     
           
            
        lc = re.sub("\s+", ",", l.strip())
        data_rows.append(lc.split(","))
    excel_out = os.path.join(this_dir, path,classification[:-3]  + "xlsx")
    column_headers = ["","","precision", "recall", "f1-score", "support"]
    df = pd.DataFrame(data_rows,columns=column_headers)
    df.to_excel(excel_out)

    









if __name__ == "__main__":
    path = "results"
    confusion = "10_confusion.txt"
    classifiction = "10_class.txt"
    
    format_confusion(path, classifiction)

