import pandas as pd

def f_importances(model, train_df):
    coef = abs(model.coef_[0])
    names = list(train_df.columns)
    
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    list_of_lists = []
    headers = ['Frame label', 'Importance']
    for name, importance in zip(names, imp):
        list_of_lists.append([name, importance])
    
    df = pd.DataFrame(list_of_lists, columns=headers)
    return df