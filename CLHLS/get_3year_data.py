import pandas as pd
import numpy as np

def get_data_between_start_to_end(df,start_year):
    # 根据输入df的起始年份得到三年后的变量后缀，如输入2002年的df，则三年后的变量后缀为 _5，输入2011年的，三年后变量后缀则为_14
    short_year = str(start_year+3)[-1] if start_year+3 < 2010 else str(start_year+3)[-2:]
    variable_list = []
    index = 0
    minux_index = len(short_year)+1
    # 把起始年份的变量名称收集记起来
    for i,variable in enumerate(df.columns):
        if variable[-minux_index:] == '_' + short_year:
            index = i
            break
        variable_list.append(variable)
     # 把三年后的变量名称收集起来，每两次访问的变量之间都有一个"dth年份_年份"变量进行隔开，所以以该变量为分割提取，遇到该变量就停止
    for j in range(index,len(df.columns)):
        if df.columns[j] == 'dth'+str(start_year+3)[-2:]+'_'+str(start_year+6)[-2:] or df.columns[j][-3:] == '_'+str(start_year+7)[-2:]:
            break
        variable_list.append(df.columns[j])
    # 根据变量名提取出当年，和三年期的问卷变量，其他年份的抛弃
    return df.loc[:,variable_list]

def get_df_peo_3years_situation(data02to05,data05to08,data08to11,data11to14):
    # data11to14里的dth11_14字段中有' '空格字符，所以该字段为字符串类型，需要将其改为数值型
    data11to14.dth11_14 = pd.to_numeric(data11to14.dth11_14,errors = 'coerce')
    # data08to11中有一位老年人的dth11_14字段中值为2011，经核对该老人为存活者，所以将其改为0
    data08to11.loc[data08to11[data08to11.dth08_11 == 2011].index,'dth08_11'] = 0

    # 整理各组数据的起始调查人数，三年存活人数，三年死亡人数，三年失访人数
    observations = []
    survivor = []
    death = []
    missing = []
    df_dict = {'df1':data02to05,'df2':data05to08,'df3':data08to11,'df4':data11to14}
    df_var_dict = {'df1':'dth02_05','df2':'dth05_08','df3':'dth08_11','df4':'dth11_14'}
    for df in df_dict.keys():
        counter = df_dict[df][df_var_dict[df]].value_counts()
        observations.append(len(df_dict[df]))
        survivor.append(counter[0])
        death.append(counter[1])
        missing.append(counter[-9])

    return pd.DataFrame({'start year observations':observations,
                     'survivor':survivor,
                      'death':death,
                      'missing':missing},index = ['2002 to 2005','2005 to 2008','2008 to 2011','2011 to 2014'])

def get_wave_data(df,start_year):
    # 需要用到的变量名
    basic_columns = ['a1','trueage','a2','f1','f41','f34','a41','a43','a51',  #personal info
                  'b21','b22','b23','b24','b25','b26','b27',                                  #psychological status
                  'd71','d72','d75','d81','d82','d86','d91','d92','d31','d32',             #living style
                  'd11b','d11c','d11d','d11e','d11f','d11g','d11h','d12',                       #social activities
                  'e1','e2','e3','e4','e5','e6',                                                     #ADL
                  'g15a1','g15b1','g15c1','g15d1','g15e1','g15f1','g15g1','g15h1','g15i1','g15j1','g15k1','g15l1','g15m1','g15n1','g15o1', #chronic diseases
                  'c11','c12','c13','c14','c15','c16','c21a','c21b','c21c','c31a','c31b','c31c','c31d','c31e','c32','c41a','c41b','c41c','c51a','c51b','c52','c53a','c53b','c53c',#CI measurement
                ]
    year_index = '_'+str(start_year+3)[-1] if start_year + 3 < 2010 else '_'+str(start_year+3)[-2:]
    
    # 根据需要用到的变量名和三年后的后缀 得到 需要用到的三年后的变量名
    ## 11年前，后续年份随访调查的年龄字段叫vage_年份； 11年后，后续年份随访调查的年龄字段叫trueage_年份；
    if start_year != 2011:
        three_years_later_columns = list(map(lambda x:x+year_index if x!='trueage' else 'vage'+year_index,basic_columns))
    else:
        three_years_later_columns = list(map(lambda x:x+year_index,basic_columns))
    # 个人信息问题只在初始年记录，后续年没有，把三年后的个人信息问题去掉
    for personal_stable_variable in list(map(lambda x:x+year_index,['a2', 'f1', 'a1', 'a41','a43'])):
        three_years_later_columns.remove(personal_stable_variable)
    total_columns = basic_columns + three_years_later_columns
    
    # 筛选3年后随访还活着且没失访的老年人
    death_index_var = 'dth'+str(start_year)[-2:]+'_'+str(start_year+3)[-2:]
    live_df = df[df[death_index_var] == 0]
    
    wave_data = live_df[total_columns]
    for column in wave_data.columns:
        wave_data[column] = pd.to_numeric(wave_data[column],errors = 'coerce')
    return wave_data

def replace_abnormal_and_fill_nan(df,start_year):
    # 以下问题的问卷选项中存在官方的取值8，或者有可能出现取值8。所以不能将这些变量的8作为缺失处理
    questions_may_contain8 = ['vage','trueage','a2','a41','f1','b21','b22','b23','b24','b25','b26','b27','d75','d86','d12','c11','c12','c13','c14','c15','c16','c21a','c21b','c21c','c31a','c31b','c31c','c31d','c31e','c32','c41a','c41b','c41c','c51a','c51b','c52','c53a','c53b','c53c']
    year_index = '_'+str(start_year+3)[-1] if start_year + 3 < 2010 else '_'+str(start_year+3)[-2:]
    later_questions_may_contain8 = list(map(lambda x:x+year_index,questions_may_contain8))
    total_questions_may_contain8 = questions_may_contain8+later_questions_may_contain8
    # 把剩下的变量中的按codebook说明的缺失取值或不合理取值变成缺失值。
    variable_needed_processed = list(set(df.columns) - set(total_questions_may_contain8))
    new_df = df.copy()
    for var in variable_needed_processed:
        new_df[var] = new_df[var].replace(8,np.nan)
        new_df[var] = new_df[var].replace(88,np.nan)
        new_df[var] = new_df[var].replace(888,np.nan)
        new_df[var] = new_df[var].replace(8888,np.nan)
        new_df[var] = new_df[var].replace(9,np.nan)
        new_df[var] = new_df[var].replace(99,np.nan)
        new_df[var] = new_df[var].replace(99,np.nan)
        new_df[var] = new_df[var].replace(9999,np.nan)
    # 用众数填补变量的缺失值
    print('Missing values situation')
    print((new_df.loc[:,new_df.isnull().any()].isnull().sum()/len(df)).apply(lambda x:str('%.2f' %(x*100))+'%'))
    for col in new_df.columns[new_df.isnull().any()]:
        mode = new_df[col].value_counts().sort_values(ascending = False).index[0]
        new_df[col].fillna(mode,inplace = True)
        
    # 去掉MMSE问题全部不能回答的
    MMSE_questions = ['c11','c12','c13','c14','c15','c21a','c21b','c21c','c31a','c31b','c31c','c31d','c31e','c32','c41a','c41b','c41c','c51a','c51b','c52','c53a','c53b','c53c']
    MMSE_questions_later = list(map(lambda x:x+year_index,MMSE_questions))
    for var in MMSE_questions:
        new_df[var] = new_df[var].replace(8,np.nan)
        new_df[var] = new_df[var].replace(88,np.nan)
        new_df[var] = new_df[var].replace(888,np.nan)
        new_df[var] = new_df[var].replace(8888,np.nan)
        new_df[var] = new_df[var].replace(9,np.nan)
        new_df[var] = new_df[var].replace(99,np.nan)
        new_df[var] = new_df[var].replace(99,np.nan)
        new_df[var] = new_df[var].replace(9999,np.nan)
    print('Samples before drop current MMSE missing : ',len(new_df))
    new_df = new_df[new_df.isnull().sum(axis = 1)<23]
    print('Samples before drop current MMSE missing : ',len(new_df))
    
    for var in MMSE_questions_later:
        new_df[var] = new_df[var].replace(8,np.nan)
        new_df[var] = new_df[var].replace(88,np.nan)
        new_df[var] = new_df[var].replace(888,np.nan)
        new_df[var] = new_df[var].replace(8888,np.nan)
        new_df[var] = new_df[var].replace(9,np.nan)
        new_df[var] = new_df[var].replace(99,np.nan)
        new_df[var] = new_df[var].replace(99,np.nan)
        new_df[var] = new_df[var].replace(9999,np.nan)
    print('Samples before drop later MMSE missing : ',len(new_df))
    new_df = new_df[new_df.isnull().sum(axis = 1)<23]
    print('Samples before drop later MMSE missing : ',len(new_df))
    
    
    return new_df

def get_period_data():
    print('Reading data ...')
    data_02to18 = pd.read_csv('./data/clhls_2002_2018_utf8.csv',low_memory=False)
    data_05to18 = pd.read_csv('./data/clhls_2005_2018_utf8.csv',low_memory=False)
    data_08to18 = pd.read_csv('./data/clhls_2008_2018_utf8.csv',low_memory=False)
    data_11to18 = pd.read_csv('./data/clhls_2011_2018_utf8.csv',low_memory=False)
    print('Getting 3 year periods data ...')
    data02to05 = get_data_between_start_to_end(data_02to18,2002)
    data05to08 = get_data_between_start_to_end(data_05to18,2005)
    data08to11 = get_data_between_start_to_end(data_08to18,2008)
    data11to14 = get_data_between_start_to_end(data_11to18,2011)
    
    situation = get_df_peo_3years_situation(data02to05,data05to08,data08to11,data11to14)
    print('Number of investgated elders situations : ','\n',situation)
    
    print('Selecting needed variables ...')
    first_wave = get_wave_data(data02to05,2002)
    second_wave = get_wave_data(data05to08,2005)
    third_wave = get_wave_data(data08to11,2008)
    fourth_wave = get_wave_data(data11to14,2011)
    print('Filling missing values ...')
    first_wave_new = replace_abnormal_and_fill_nan(first_wave,2002)
    second_wave_new = replace_abnormal_and_fill_nan(second_wave,2005)
    third_wave_new = replace_abnormal_and_fill_nan(third_wave,2008)
    fourth_wave_new = replace_abnormal_and_fill_nan(fourth_wave,2011)
    
    print('Before drop age<60 : ')
    print('Wave 1: ',len(first_wave_new))
    print('Wave 2: ',len(second_wave_new))
    print('Wave 3: ',len(third_wave_new))
    print('Wave 4: ',len(fourth_wave_new))
    first_wave_new = first_wave_new[first_wave_new.trueage>=60]
    second_wave_new = second_wave_new[second_wave_new.trueage>=60]
    third_wave_new = third_wave_new[third_wave_new.trueage>=60]
    fourth_wave_new = fourth_wave_new[fourth_wave_new.trueage>=60]
    print('After drop age<60 : ')
    print('Wave 1: ',len(first_wave_new))
    print('Wave 2: ',len(second_wave_new))
    print('Wave 3: ',len(third_wave_new))
    print('Wave 4: ',len(fourth_wave_new))
    
    
    return first_wave_new,second_wave_new,third_wave_new,fourth_wave_new



