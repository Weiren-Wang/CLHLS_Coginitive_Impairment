import pandas as pd
import numpy as np


#####################################        calculate CI,ADL,depression scores and get label      ###############################################
def calculate_CI_score(x,periods = 'current',start_year = None):
    if periods == 'current':
        questions = ['c11','c12','c13','c14','c15','c16','c21a','c21b','c21c','c31a','c31b','c31c','c31d','c31e','c32','c41a','c41b','c41c','c51a','c51b','c52','c53a','c53b','c53c']
    elif periods == 'three_years_later':
        year_index = '_'+str(start_year+3)[-1] if start_year + 3 < 2010 else '_'+str(start_year+3)[-2:]
        questions = list(map(lambda x:x+year_index,['c11','c12','c13','c14','c15','c16','c21a','c21b','c21c','c31a','c31b','c31c','c31d','c31e','c32','c41a','c41b','c41c','c51a','c51b','c52','c53a','c53b','c53c']))
    
    CI_score = 0
    for question in questions:
        if question[0:3] != 'c16': #如果不是数食物的题
            if x[question] == 1:CI_score += 1
        else: #如果是数食物的题
            food_score = x[question] if x[question]<=7 else 7
            CI_score += food_score
    return CI_score

def get_CI_label(x,periods):
    if periods == 'current':
        return 0 if x.current_CI_score>= 18 else 1
    if periods == 'three_years_later':
        return 0 if x.later_CI_score>= 18 else 1
    
    
def calculate_depression_score(x,periods = 'current',start_year = None):
    if periods == 'current':
        questions = ['b21','b22','b23','b24','b25','b26','b27']
    elif periods == 'three_years_later':
        year_index = '_'+str(start_year+3)[-1] if start_year + 3 < 2010 else '_'+str(start_year+3)[-2:]
        questions = list(map(lambda x:x+year_index,['b21','b22','b23','b24','b25','b26','b27']))
    
    depression_score = 0
    missing_nums = 0
    # 只留下所有问题都回答了的样本
    for question in questions:
        if x[question] != 8:#如果能回答
            depression_score += x[question]
        else: #如果不能回答
            missing_nums += 1
    
    return depression_score if missing_nums == 0 else np.nan


def calculate_ADL_label(x,periods = 'current',start_year = None):
    if periods == 'current':
        questions = ['e1','e2','e3','e4','e5','e6']
    elif periods == 'three_years_later':
        year_index = '_'+str(start_year+3)[-1] if start_year + 3 < 2010 else '_'+str(start_year+3)[-2:]
        questions = list(map(lambda x:x+year_index,['e1','e2','e3','e4','e5','e6']))
    
    ADL_score = 0
    for question in questions:
        if x[question] == 1:
            continue
        else:
            ADL_score += 1
    
    return 0 if ADL_score == 0 else 1

def get_scores(df,start_year):
    ## CI score
    new_df = df.copy()
    current_questions = ['c11','c12','c13','c14','c15','c16','c21a','c21b','c21c','c31a','c31b','c31c','c31d','c31e','c32','c41a','c41b','c41c','c51a','c51b','c52','c53a','c53b','c53c']
    year_index = '_'+str(start_year+3)[-1] if start_year + 3 < 2010 else '_'+str(start_year+3)[-2:]
    three_year_questions = list(map(lambda x:x+year_index,['c11','c12','c13','c14','c15','c16','c21a','c21b','c21c','c31a','c31b','c31c','c31d','c31e','c32','c41a','c41b','c41c','c51a','c51b','c52','c53a','c53b','c53c']))
    questions = current_questions + three_year_questions
#     for column in questions:
#         new_df[column] = pd.to_numeric(new_df[column],errors = 'coerce')
    
    new_df['current_CI_score'] = new_df.apply(lambda x:calculate_CI_score(x),axis = 1)
    new_df['later_CI_score'] = new_df.apply(lambda x:calculate_CI_score(x,periods = 'three_years_later',start_year = start_year),axis = 1)
    new_df.drop(questions,axis = 1,inplace = True)
    
    new_df['current_CI'] = new_df.apply(lambda x:get_CI_label(x,'current'),axis = 1)
    new_df['later_CI'] = new_df.apply(lambda x:get_CI_label(x,'three_years_later'),axis = 1)
    ## depression score
    current_questions = ['b21','b22','b23','b24','b25','b26','b27']
    three_year_questions = list(map(lambda x:x+year_index,['b21','b22','b23','b24','b25','b26','b27']))
    questions = current_questions + three_year_questions
    for column in questions:
        new_df[column] = pd.to_numeric(new_df[column],errors = 'coerce')
    
    new_df['depression_score'] = new_df.apply(lambda x:calculate_depression_score(x),axis = 1)
    new_df['depression_score_later'] = new_df.apply(lambda x:calculate_depression_score(x,periods = 'three_years_later',start_year = start_year),axis = 1)
    
    print('Before drop Depression missing : ',len(new_df))
    new_df.drop(questions,axis = 1,inplace = True)
    new_df.dropna(subset = ['depression_score','depression_score_later'],axis = 0,inplace = True)
    print('After drop Depression missing : ',len(new_df))
    ## ADL score
    current_questions = ['e1','e2','e3','e4','e5','e6']
    three_year_questions = list(map(lambda x:x+year_index,['e1','e2','e3','e4','e5','e6']))
    questions = current_questions + three_year_questions
    for column in questions:
        new_df[column] = pd.to_numeric(new_df[column],errors = 'coerce')
    
    new_df['ADL'] = new_df.apply(lambda x:calculate_ADL_label(x),axis = 1)
    new_df['ADL_later'] = new_df.apply(lambda x:calculate_ADL_label(x,periods = 'three_years_later',start_year = start_year),axis = 1)
    
    new_df.drop(questions,axis = 1,inplace = True)
    return new_df

###################################################        process features      #########################################################
# 教育程度处理 0年，0-6年，7年及以上
def get_education(x):
    if x == 0:return 0
    elif x<=6:return 1
    else:return 2
    
def process_province(x):
    # 把出生在国外的和回答不知道的归为一类
    if x in np.arange(90,100):
        return 100
    else:
        return x
    
def process_personal_invariant_variable(df):
    # 民族处理，汉 vs 非汉
    print('Processing invariant variables ...')
    df['nation_group'] = df.a2.apply(lambda x:0 if x== 1 else 1)
    del df['a2']
    # 教育程度处理，0,1-6,7以上
    df['education'] = df.f1.apply(lambda x:get_education(x))
    del df['f1']
    df['borned_prov'] = df.a41.apply(lambda x:process_province(x))
    del df['a41']
    # 性别改为0女，1男
    df['a1'] = df['a1'].apply(lambda x:0 if x==2 else 1)
    # 户口类型改为0城镇，1农村
    df['a43'] = df['a43'].apply(lambda x:0 if x==1 else 1)
    print('Done!')
    
    
# 年龄处理
def get_agegroup(x):
    if x<=75 : return 0
    elif x<=85 : return 1
    elif x<=95 : return 2
    else:return 3

# 婚姻处理
def get_martial_status(x):
    if x==1 : return 1
    elif x==2 : return 1
    else:return 0

# 经济情况处理
def get_economy_status(x):
    if x==3 : return  0# normal 
    elif x==1 : return 1 # rich
    elif x==2 : return 1 # rich
    else: return 2 #poor

# 同居情况处理
def get_coresidence(x):
    if x==2:return 0 #独居
    if x==1:return 1 #和家人
    if x==3:return 2#养老院
    
def process_personal_variant_variable(df,start_year):
    print('Processing variant variables ...')
    year_index = '_'+str(start_year+3)[-1] if start_year + 3 < 2010 else '_'+str(start_year+3)[-2:]
    # 根据年龄得到年龄层
    age_variable = 'vage' if start_year != 2011 else 'trueage'
    age_variable += year_index
    later_agegroup = 'agegroup'+year_index
    df['agegroup'] = df.trueage.apply(lambda x:get_agegroup(x))
    df[later_agegroup] = df[age_variable].apply(lambda x:get_agegroup(x))
    del df['trueage']
    del df[age_variable]
    
    # 婚姻情况：五类归为两类：有配偶、无配偶
    later_martial_variable = 'f41'+year_index
    processed_later_martial_variable_name = 'martial_status'+year_index
    df['martial_status'] = df.f41.apply(lambda x:get_martial_status(x))
    df[processed_later_martial_variable_name] = df[later_martial_variable].apply(lambda x:get_martial_status(x))
    del df['f41']
    del df[later_martial_variable]

    # 经济情况：五类归位三类：富、一般、穷
    later_economy_variable = 'f34'+year_index
    processed_later_economy_variable_name = 'economy_status'+year_index
    df['economy_status'] = df.f34.apply(lambda x:get_economy_status(x))
    df[processed_later_economy_variable_name] = df[later_economy_variable].apply(lambda x:get_economy_status(x))
    del df['f34']
    del df[later_economy_variable]
    
    # 同居情况：修改赋值
    later_coresi_variable = 'a51'+year_index
    df['a51'] = df.a51.apply(lambda x:get_coresidence(x))
    df[later_coresi_variable] = df[later_coresi_variable].apply(lambda x:get_coresidence(x))
    
    # 生活习惯：抽烟、喝酒、运动：修改赋值 0=不，1=是
    living_style = ['d71','d81','d91']
    later_living_style = list(map(lambda x:x+year_index,living_style))
    total_living_style = living_style+later_living_style
    for col in total_living_style:
        df[col] = df[col].apply(lambda x:1 if x==1 else 0)
        
    # 水果蔬菜：修改赋值,0 = 不吃，3= 天天吃
    eat = ['d31','d32']
    later_eat = list(map(lambda x:x+year_index,eat))
    total_eat = eat+later_eat
    for col in total_eat:
        df[col] = df[col].apply(lambda x:4-x)
        
    # 社会娱乐生活：修改赋值，4 = 天天参加，0等于不参加
    life = ['d11b','d11c','d11d','d11e','d11f','d11g','d11h']
    later_life = list(map(lambda x:x+year_index,life))
    total_life = life+later_life
    for col in total_life:
        df[col] = df[col].apply(lambda x:5-x)
    
    # 慢性病处理：把回答不知道的当作没有,有病为1，没病为0
    chronic_var = ['g15a1','g15b1','g15c1','g15d1','g15e1','g15f1','g15g1','g15h1','g15i1','g15j1','g15k1','g15l1','g15m1','g15n1','g15o1']
    later_chronic_var = list(map(lambda x:x+year_index,chronic_var))
    total_var = chronic_var+later_chronic_var
    for col in total_var:
        df[col] = df[col].replace(3,2)
        df[col] = df[col].apply(lambda x:1 if x==1 else 0)
        
    print('Before drop Dementia : ',len(df))
    later_g15o1 = 'g15o1'+year_index
    df.drop(df[df.g15o1 == 1].index,axis = 0,inplace = True)
    df.drop(df[df[later_g15o1] == 1].index,axis = 0,inplace = True)
    print('After drop Dementia : ',len(df))
    df.reset_index(drop = True,inplace = True)
    print('Done!')

def process_wave_data(first_wave_new,second_wave_new,third_wave_new,fourth_wave_new):
    wave1 = get_scores(first_wave_new,2002)
    wave2 = get_scores(second_wave_new,2005)
    wave3 = get_scores(third_wave_new,2008)
    wave4 = get_scores(fourth_wave_new,2011)

    process_personal_invariant_variable(wave1)
    process_personal_invariant_variable(wave2)
    process_personal_invariant_variable(wave3)
    process_personal_invariant_variable(wave4)

    process_personal_variant_variable(wave1,2002)
    process_personal_variant_variable(wave2,2005)
    process_personal_variant_variable(wave3,2008)
    process_personal_variant_variable(wave4,2011)
    
    return wave1,wave2,wave3,wave4
