import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from  scipy.stats import chi2_contingency
import numpy as np
from scipy.stats import ttest_ind

def get_var_describe_test(df,var):
    population = list(df.groupby(var).count().iloc[:,1].sort_index())
    population_rate = list(map(lambda x:str(x)+'('+str('{:.2f}'.format(x/sum(population)*100))+'%'+')',population))
    value_index = list(df.groupby(var).count().iloc[:,1].sort_index().index)

    current_CI_count = df.groupby([var,'current_CI']).count().iloc[:,1]
    CI_now_index = list(map(lambda x:True if x[1] == 1 else False,current_CI_count.index))
    not_CI_now_index = list(map(lambda x:True if x[1] == 0 else False,current_CI_count.index))
    CI_now_population = list(current_CI_count[CI_now_index])
    not_CI_now_population = list(current_CI_count[not_CI_now_index])
    
    for i,value in enumerate(not_CI_now_population):
        if value/population[i]  == 1:# 如果某组去都没有CI，为了保持列表长度相同，在CI组补0
            CI_now_population.insert(i,0)
        not_CI_now_population[i] = str(not_CI_now_population[i])+'('+str('{:.2f}'.format(not_CI_now_population[i]/population[i]*100))+'%'+')'
    for i,value in enumerate(CI_now_population):
        CI_now_population[i] = str(CI_now_population[i])+'('+str('{:.2f}'.format(CI_now_population[i]/population[i]*100))+'%'+')'
    

    later_CI_count = df.groupby([var,'later_CI']).count().iloc[:,1]
    CI_later_index = list(map(lambda x:True if x[1] == 1 else False,later_CI_count.index))
    not_CI_later_index = list(map(lambda x:True if x[1] == 0 else False,later_CI_count.index))
    CI_later_population = list(later_CI_count[CI_later_index])
    not_CI_later_population = list(later_CI_count[not_CI_later_index])
    
    for i,value in enumerate(not_CI_later_population):
        if value/population[i]  == 1:# 如果某组去都没有CI，为了保持列表长度相同，在CI组补0
            CI_later_population.insert(i,0)
        not_CI_later_population[i] = str(not_CI_later_population[i])+'('+str('{:.2f}'.format(not_CI_later_population[i]/population[i]*100))+'%'+')'
    
    for i,value in enumerate(CI_later_population):
        CI_later_population[i] = str(CI_later_population[i])+'('+str('{:.2f}'.format(CI_later_population[i]/population[i]*100))+'%'+')'
    

    return pd.DataFrame({'value':value_index,'population':population_rate,'CI now':CI_now_population,'not CI now':not_CI_now_population,
                          'CI 3 years later':CI_later_population,'not CI 3 years later':not_CI_later_population})


def get_total_descriptive_df(df):
    var_list = ['agegroup','a1','nation_group','education','martial_status','economy_status','a43','a51',  #personal info
                  'd71','d81','d91','d31','d32',             #living style
                  'd11b','d11c','d11d','d11e','d11f','d11g','d11h',                       #social activities
                  'ADL',                                                   #ADL
                  'g15a1','g15b1','g15c1','g15d1','g15e1','g15f1','g15g1','g15h1','g15i1','g15j1','g15k1','g15l1','g15m1','g15n1','g15o1', #chronic disease
                ]
    df_out = pd.DataFrame(columns = ['variable','value', 'population', 'CI now', 'not CI now', 'CI 3 years later','not CI 3 years later'])
    for var in var_list:
        df_var = get_var_describe_test(df,var)
        df_var['variable'] = var
        df_var = df_var[['variable','value', 'population', 'CI now', 'not CI now', 'CI 3 years later','not CI 3 years later']]
        df_out = pd.concat([df_out,df_var],axis = 0)
    return df_out

def get_total_df_description(total_df):
    var_list = ['agegroup','nation_group','education','martial_status','economy_status','ADL','std_plant_flower_bird','std_read_news_books','std_raise_domestic_animals','std_majong_or_cards','std_tv_or_radio','std_eat_fruits','std_eat_vegs']
    rename_dict_base = {'a1':'gender','a43':'residence','a51':'co-habitation','d71':'smoke','d81':'alcohol','d91':'exercise','d31':'eat_fruits','d32':'eat_vegs','d11b':'outdoor_activity','d11c':'plant_flower_bird','d11d':'read_news_books',
                          'd11e':'raise_domestic_animals','d11f':'majong_or_cards','d11g':'tv_or_radio','d11h':'social_activity',
                          'g15a1':'hypertension','g15b1':'diabetes','g15c1':'heart_disease','g15d1':'stoke_CVD','g15e1':'trachea_or_lung',
                          'g15f1':'tuberculosis','g15g1':'cataracts','g15h1':'glaucoma','g15i1':'cancer','g15j1':'prostate','g15k1':'stomach_ulcer',
                          'g15l1':'Parkinson','g15m1':'bedsore','g15n1':'arthritis'}
    descrptive_vars = var_list + list(rename_dict_base.values())
    descrptive_vars = list(set(descrptive_vars) & set(total_df.columns))
    df_out = pd.DataFrame(columns = ['variable','value', 'population', 'CI now', 'not CI now', 'CI 3 years later','not CI 3 years later'])
    for var in descrptive_vars:
        df_var = get_var_describe_test(total_df,var)
        df_var['variable'] = var
        df_var = df_var[['variable','value', 'population', 'CI now', 'not CI now', 'CI 3 years later','not CI 3 years later']]
        df_out = pd.concat([df_out,df_var],axis = 0)
    return df_out


def get_categorical_describe(total_df,var):
    grouped = total_df.groupby(['later_CI',var]).count().iloc[:,1]
    not_CI = grouped[list(map(lambda x:True if x[0]==0 else False,list(grouped.index)))]
    CI = grouped[list(map(lambda x:True if x[0]==1 else False,list(grouped.index)))]
    res_notCI = [0]*len(total_df[var].unique())
    res_CI = [0]*len(total_df[var].unique())
    for i in range(len(not_CI)):
        res_notCI[i] = str(not_CI.iloc[i])+ '(' + str('%.2f' %(not_CI.iloc[i]/(not_CI.iloc[i]+CI.iloc[i])*100)) +')'
        res_CI[i] = str(CI.iloc[i]) + '(' + str('%.2f' %(CI.iloc[i]/(not_CI.iloc[i]+CI.iloc[i])*100)) + ')'
    df = pd.concat([pd.DataFrame(res_notCI),pd.DataFrame(res_CI)],axis = 1)
    df.columns = ['not_CI','CI']
    return df

def get_numerical_describe(total_df,var):
    mean_ = total_df.groupby('later_CI').mean()[var]
    std_ = total_df.groupby('later_CI').std()[var]
    res = [0]*len(mean_)
    for i in range(len(mean_)):
        res[i] = str('%.4f' %mean_[i]) + '(' + str('%.4f' %std_[i]) + ')'
    
    return pd.DataFrame({'not_CI':res[0],'CI':res[1]},index = ['Mean(SD)'])

def get_pvalues(total_df):
    var_list = ['agegroup','nation_group','education','martial_status','economy_status','ADL']
    rename_dict_base = {'a1':'gender','a43':'residence','a51':'co-habitation','d71':'smoke','d81':'alcohol','d91':'exercise','d31':'eat_fruits','d32':'eat_vegs','d11b':'outdoor_activity','d11c':'plant_flower_bird','d11d':'read_news_books',
                          'd11e':'raise_domestic_animals','d11f':'majong_or_cards','d11g':'tv_or_radio','d11h':'social_activity',
                          'g15a1':'hypertension','g15b1':'diabetes','g15c1':'heart_disease','g15d1':'stoke_CVD','g15e1':'trachea_or_lung',
                          'g15f1':'tuberculosis','g15g1':'cataracts','g15h1':'glaucoma','g15i1':'cancer','g15j1':'prostate','g15k1':'stomach_ulcer',
                          'g15l1':'Parkinson','g15m1':'bedsore','g15n1':'arthritis'}
    descrptive_vars = var_list + list(rename_dict_base.values())
    numerical_vars = ['depression_score','tour_times']
    var_name = []
    pvalues = []
    for var in descrptive_vars:
        kf_data = np.array(total_df.groupby(['later_CI',var]).count().iloc[:,0]).reshape(2,len(total_df[var].unique()))
        kf = chi2_contingency(kf_data)
        var_name.append(var)
        pvalues.append(kf[1])
    for var in numerical_vars:
        a = total_df[total_df.later_CI == 0][var]
        b = total_df[total_df.later_CI == 1][var]
        t_res = ttest_ind(a, b)
        var_name.append(var)
        pvalues.append(t_res.pvalue)
    df_out = pd.DataFrame({'Variable':var_name,'p-values':pvalues})
    return df_out