import statsmodels.api as sm
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm

def logistic_regression(input_df,label_column,categorical_variable_list=[],CI_alpha=0.05,standardized_diff = False):
    df = input_df.copy()
    if standardized_diff:
        diff_var = list(filter(lambda x:True if x[-4:] == 'diff' else False,df.columns))
        if diff_var:
            print('Standardizing diff variables ...')
            for var in diff_var:
                df[var] = df[var].apply(lambda x:x if x<= 0 else 1)
                df[var] = df[var].apply(lambda x:x if x>=0 else -1)
            print('Standardize diff variables done!')
    
    # 一起逻辑回归
    x_col = [column for column in df.columns if label_column not in column and column not in categorical_variable_list]
    y_col = [column for column in df.columns if column == label_column]
    # 
    if 'borned_prov' in x_col:x_col.remove('borned_prov')
    if 'agegroup_3years_later' in x_col:x_col.remove('agegroup_3years_later')
    if 'agegroup_diff' in x_col:x_col.remove('agegroup_diff')
 
    x = np.array(df[x_col])
    if categorical_variable_list:
        onehot_cat_var = pd.get_dummies(df[categorical_variable_list[0]],prefix = categorical_variable_list[0])
        drop_ref_col = list(filter(lambda x:True if x[-2:] == '_0' or x[-4:] == '_0.0' else False,onehot_cat_var.columns))
        onehot_cat_var.drop(drop_ref_col,axis = 1,inplace = True)
        for variable in categorical_variable_list[1:]:
            tmp = pd.get_dummies(df[variable],prefix = variable)
            drop_ref_col = list(filter(lambda x:True if x[-2:] == '_0' or x[-4:] == '_0.0' else False,tmp.columns))
            tmp.drop(drop_ref_col,axis = 1,inplace = True)
            onehot_cat_var = pd.concat([onehot_cat_var,tmp],axis = 1)
        onehot_cat_var_values = np.array(onehot_cat_var)
        x = np.hstack([x,onehot_cat_var_values])
    y = np.array(df[label_column].astype(float)).reshape(-1)
    model = sm.Logit(y,x.astype(float)).fit()
    para = pd.DataFrame(np.exp(model.conf_int(CI_alpha)),columns = ['95%lower','95%upper'])
    para['coef'] = model.params
    para['OddRatio'] = np.exp(para.coef)
    x_name = x_col.copy()
    if categorical_variable_list:
        x_name.extend(onehot_cat_var.columns)
    para['Variable'] = x_name
    para['pvalues'] = model.pvalues

    var_value_count = []
    for vari in x_name:
        if vari[-1] == '0':
            if vari[-4] == '-':
                var_value = float(vari[-4:])
                var_name = vari[:-5]
                print(vari)
                var_value_count.append(len(df[df[var_name]==var_value]))
            else:
                var_value = float(vari[-3:])
                var_name = vari[:-4]
                var_value_count.append(len(df[df[var_name]==var_value]))
        else:
            var_value_count.append(len(df[df[vari]==1]))
    para['Num of samples'] = var_value_count
    para = para[['Variable','Num of samples','coef','OddRatio','95%lower','95%upper','pvalues']]
    std_x = np.array((df[x_col]-np.mean(df[x_col]))/np.std(df[x_col]))
    if categorical_variable_list:
        std_x = np.hstack([std_x,onehot_cat_var_values])
    model2 = sm.Logit(y,std_x.astype(float)).fit()
    importance_df = pd.DataFrame(para['Variable'])
    importance_df['coef'] = model2.params
    importance_df.sort_values(by = 'coef',ascending=False,key=np.abs,inplace = True)
    importance_df['importance'] = range(1,len(importance_df)+1)
    importance_df.drop('coef',axis=1,inplace=True)
    importance_df.set_index('importance',inplace = True)
    
#     # 单个变量回归
#     single_para = pd.DataFrame(columns = ['Variable','solo_coef','solo_OddRatio','solo_95%lower','solo_95%upper','solo_pvalues'])
#     for j in range(x.shape[1]):
#         single_x = x[:,j]
#         variable = x_name[j]
#         single_model = sm.Logit(y,single_x.astype(float)).fit()
#         solo_coef = single_model.params
#         solo_OR = np.exp(single_model.params)
#         solo_lower = np.exp(single_model.conf_int(CI_alpha)[0][0])
#         solo_upper = np.exp(single_model.conf_int(CI_alpha)[0][1])
#         solo_pvalue = single_model.pvalues
#         temp = pd.DataFrame({'Variable':variable,'solo_coef':solo_coef,'solo_OddRatio':solo_OR,'solo_95%lower':solo_lower,'solo_95%upper':solo_upper,'solo_pvalues':solo_pvalue})
#         single_para = pd.concat([single_para,temp],axis = 0)
#     para = pd.merge(para,single_para,on = 'Variable')  
                                
    res = {'summary':model.summary(),
       'params':para,
       'feature_importance':importance_df}
    return res