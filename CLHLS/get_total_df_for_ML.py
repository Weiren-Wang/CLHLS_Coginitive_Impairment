from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def get_training_df(df1,df2,df3,df4):
    drop_list = ['d72','d75','d82','d86','d92'] # 用不到的变量
    rename_dict_base = {'a1':'gender','a43':'residence','a51':'co-habitation','d71':'smoke','d81':'alcohol','d91':'exercise','d31':'eat_fruits','d32':'eat_vegs','d11b':'outdoor_activity','d11c':'plant_flower_bird','d11d':'read_news_books',
                          'd11e':'raise_domestic_animals','d11f':'majong_or_cards','d11g':'tv_or_radio','d11h':'social_activity','d12':'tour_times',
                          'g15a1':'hypertension','g15b1':'diabetes','g15c1':'heart_disease','g15d1':'stoke_CVD','g15e1':'trachea_or_lung',
                          'g15f1':'tuberculosis','g15g1':'cataracts','g15h1':'glaucoma','g15i1':'cancer','g15j1':'prostate','g15k1':'stomach_ulcer',
                          'g15l1':'Parkinson','g15m1':'bedsore','g15n1':'arthritis','g15o1':'dementia'} # 把问卷题目代码改为变量名
    later_year = [2005,2008,2011,2014]
    
    for df,year in zip([df1,df2,df3,df4],later_year):
        year_index = '_'+str(year)[-1] if year<2011 else '_'+str(year)[-2:]
        drop_list_later = list(map(lambda x:x+year_index,drop_list))
        rename_dict = rename_dict_base.copy()
        for key in rename_dict_base.keys():
            later_key = key+year_index
            rename_dict[later_key] = rename_dict[key]+'_3years_later'

        df.drop(drop_list+drop_list_later,axis = 1,inplace = True)
        df.rename(columns = rename_dict,inplace = True)
        
        ind = -2 if year < 2011 else -3
        for col in df.columns:
            if col[ind:] == year_index:
                df.rename(columns = {col:col[:ind]+'_3years_later'},inplace = True)
    print(len(df1),len(df2),len(df3),len(df4))
    return pd.concat([df1,df2,df3,df4],axis = 0)

def get_total_df(wave1,wave2,wave3,wave4):
    df1 = wave1.copy()
    df2 = wave2.copy()
    df3 = wave3.copy()
    df4 = wave4.copy()

    total_df = get_training_df(df1,df2,df3,df4)
    print(len(total_df))


    # 标准化数值型变量
    scaler1,scaler2,scaler3,scaler4 = StandardScaler(),StandardScaler(),StandardScaler(),StandardScaler()
    total_df['depression_score'] = scaler1.fit_transform(np.array(total_df['depression_score']).reshape(-1,1))
    total_df['depression_score_later'] = scaler2.fit_transform(np.array(total_df['depression_score_later']).reshape(-1,1))
    total_df['tour_times'] = scaler3.fit_transform(np.array(total_df['tour_times']).reshape(-1,1))
    total_df['tour_times_3years_later'] = scaler4.fit_transform(np.array(total_df['tour_times_3years_later']).reshape(-1,1))

    # 为了提高precision，对抗样本不平衡的特别特征工程
    total_df['std_eat_fruits'] = total_df['eat_fruits'].apply(lambda x:0 if x == 0 else 1)
    total_df['std_eat_fruits_3years_later'] = total_df['eat_fruits'].apply(lambda x:0 if x == 0 else 1)
    total_df['std_eat_vegs'] = total_df['eat_vegs'].apply(lambda x:0 if x==0 or x==1 else 1)
    total_df['std_eat_vegs_3years_later'] = total_df['eat_vegs'].apply(lambda x:0 if x==0 or x==1 else 1)
    total_df['std_outdoor_activity'] = total_df['outdoor_activity'].apply(lambda x:1 if x == 4 else 0)
    total_df['std_outdoor_activity_3years_later'] = total_df['outdoor_activity'].apply(lambda x:1 if x == 4 else 0)
    total_df['std_plant_flower_bird'] = total_df['plant_flower_bird'].apply(lambda x:0 if x==0 or x==1 else 1)
    total_df['std_plant_flower_bird_3years_later'] = total_df['plant_flower_bird'].apply(lambda x:0 if x==0 or x==1 else 1)
    total_df['std_read_news_books'] = total_df['read_news_books'].apply(lambda x:0 if x==0 else 1)
    total_df['std_read_news_books_3years_later'] = total_df['read_news_books'].apply(lambda x:0 if x==0 else 1)
    total_df['std_raise_domestic_animals'] = total_df['raise_domestic_animals'].apply(lambda x:1 if x==4 else 0)
    total_df['std_raise_domestic_animals_3years_later'] = total_df['raise_domestic_animals'].apply(lambda x:1 if x==4 else 0)
    total_df['std_majong_or_cards'] = total_df['majong_or_cards'].apply(lambda x:1 if x==4 else 0)
    total_df['std_majong_or_cards_3years_later'] = total_df['majong_or_cards'].apply(lambda x:1 if x==4 else 0)
    total_df['std_tv_or_radio'] = total_df['tv_or_radio'].apply(lambda x:0 if x==0 else 1)
    total_df['std_tv_or_radio_3years_later'] = total_df['tv_or_radio'].apply(lambda x:0 if x==0 else 1)
    # 去掉不显著的变量
    del total_df['social_activity']
    del total_df['social_activity_3years_later']
    total_df.drop(['heart_disease','trachea_or_lung','tuberculosis','cataracts','glaucoma','cancer','prostate','stomach_ulcer','bedsore','arthritis'],axis = 1,inplace =True)
    total_df.drop(['heart_disease_3years_later','trachea_or_lung_3years_later','tuberculosis_3years_later','cataracts_3years_later','glaucoma_3years_later','cancer_3years_later','prostate_3years_later','stomach_ulcer_3years_later','bedsore_3years_later','arthritis_3years_later'],axis = 1,inplace =True)
    
    
    # 构建变量：CI分数变化值
    total_df['CI_diff'] = total_df['later_CI_score'] - total_df['current_CI_score']
    return total_df
