import numpy as np
import pandas as pd

def get_current_diff_variable_list(total_df):
    current_variables = []
    later_variables = []
    for col in total_df.columns:
        if col[-12:] == '3years_later':
            later_variables.append(col)
        if col[-12:] != '3years_later':
            current_variables.append(col)

    for var in ['current_CI_score',
                 'later_CI_score',
                 'current_CI',
                 'later_CI',
                 'depression_score_later',
                 'ADL_later',
                 'CI_diff'
                 ]:
        current_variables.remove(var)

    later_variables.extend(['ADL_later','depression_score_later'])
    person_basic_variables = ['gender','residence','education','borned_prov','nation_group']

    current_variables_without_personal = current_variables.copy()
    for var in person_basic_variables:
        current_variables_without_personal.remove(var)

    current_variables_without_personal.sort()
    later_variables.sort()

    print('Creating diff variables based on current variables and 3 years later variables ...')
    for x,y in zip(current_variables_without_personal,later_variables):
        total_df[x+'_diff'] = total_df[y] - total_df[x]
    print('Done!')
        
    diff_var = list(map(lambda x:x+'_diff',current_variables_without_personal))
    current_variables.remove('dementia')
    diff_var.remove('dementia_diff')
    return current_variables,diff_var