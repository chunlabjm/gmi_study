import os
import sys

import sbm
import numpy as np
import pandas as pd
import json
import itertools as it
import sklearn
import scipy
import seaborn as sns
import joblib 

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

if __name__ == "__main__":
    print("test")

def generate_bio_traits(mtp_data, mtp_meta):
    
    # profile table : species, class, family
    all_table_sp = sbm.mtp.util.profile(mtp_data, 'species')
    all_table_cl = sbm.mtp.util.profile(mtp_data,'class')
    all_table_fa = sbm.mtp.util.profile(mtp_data,'family')

    n_sample = all_table_sp.shape[0]
    id_vars = mtp_meta['_id']
    study_uids = mtp_meta['study_uid']
    host_categories = mtp_meta['host_category']
    total_read_cnt = sbm.mtp.util.num_reads(mtp_data)

    # subsampling to normalize various sample sizes
    subsampling_table = sbm.mtp.util.random_subsampling(all_table_sp,n=5000, replace=True,seed = 0)
    
    # 1. calculate alpha diversity
    print("1. alpha")
    profile_table = subsampling_table.copy()
    #profile_table = all_table_sp.copy()
    alpha_df = pd.DataFrame({
                          '_id' : id_vars,
                          'shannon' : sbm.mtp.alpha.shannon(profile_table),
                          'otus' : sbm.mtp.alpha.numotu(profile_table),
                          'invsimpson' : sbm.mtp.alpha.invsimpson(profile_table),
                          'shannon_eq' : sbm.mtp.alpha.shannon_e(profile_table),
                          'berger_parker_d' : sbm.alpha.alpha_diversity('berger_parker_d',profile_table),
                          'goods' : 1 - np.sum(profile_table == 1, axis=1) / total_read_cnt,
                          'pd' : sbm.func.div_pd(mtp_data),
                          'total_read_cnt' : total_read_cnt},
                          index = mtp_meta.index.values)
    print("1. alpha...completed")
    
    # 2. calculate beta diversity
    print("2. beta")

    # rpy 성능때문에 프로세스 안에서 r 실행
    if n_sample > 3000:
        print("Because sample size > 3000, you should execute R in other kernel.")
    else:
        print("Because sample size <= 3000, you should execute R in other kernel.")

    d_jsd = sbm.mtp.beta.jsd(profile_table)
    d_bc = sbm.mtp.beta.pdist(profile_table)

    # r 에서 읽기 위한 distance input, study uid, csv 파일로 저장 (condition은 study효과를 없앤 beta diversity값을 구하기 위해)
    pd.DataFrame(d_jsd,columns=id_vars,index=id_vars).to_csv("jsd.csv",index=True,header=True)
    pd.DataFrame(d_bc,columns=id_vars,index=id_vars).to_csv("bc.csv",index=True,header=True)
    study_uids.to_csv("study_uid_for_condition.csv")

    # r 실행
    os.system('Rscript r_scripts/dbRDA_for_GMI.R jsd.csv bc.csv study_uid_for_condition.csv')
    
    # 결과파일 load
    dbrda_coord_jsd = pd.read_csv("dbrda_coord_jsd.csv",header = 0, index_col = 0)
    dbrda_coord_bc = pd.read_csv("dbrda_coord_bc.csv",header = 0, index_col = 0)
    
    dbrda_coord_jsd = dbrda_coord_jsd.assign(_id = dbrda_coord_jsd.index)
    dbrda_coord_bc = dbrda_coord_bc.assign(_id = dbrda_coord_bc.index)

    print("2. beta...completed")
    
    # 3. Dissimilarity healthy and diseased group (Anna karenina effects)
    print("3. Dissimilarity")
    healthy_avg_dist_jsd = [0]*len(d_jsd)
    healthy_avg_dist_bc = [0]*len(d_jsd)
    dysbiosis_avg_dist_jsd = [0]*len(d_jsd)
    dysbiosis_avg_dist_bc = [0]*len(d_jsd)
    healthy_avg_dist_jsd_in = [0]*len(d_jsd)
    healthy_avg_dist_bc_in = [0]*len(d_jsd)
    dysbiosis_avg_dist_jsd_in = [0]*len(d_jsd)
    dysbiosis_avg_dist_bc_in = [0]*len(d_jsd)

    # calculate distance with other node
    for i in range(len(host_categories)):
        other_idx = list(set(range(len(d_jsd[i]))) - set([i]))
        tmp_category = host_categories[other_idx]
        tmp_study_uid = study_uids[other_idx]
        my_study = study_uids[i]

        tmp_d_jsd = d_jsd[i][other_idx]
        tmp_d_bc = d_bc[i][other_idx]

        healthy_avg_dist_jsd[i] = np.mean(tmp_d_jsd[tmp_category == "healthy"])
        healthy_avg_dist_bc[i] = np.mean(tmp_d_bc[tmp_category == "healthy"])
        dysbiosis_avg_dist_jsd[i] = np.mean(tmp_d_jsd[tmp_category == "diseased"])
        dysbiosis_avg_dist_bc[i] = np.mean(tmp_d_bc[tmp_category == "diseased"])
        healthy_avg_dist_jsd_in[i] = np.mean(tmp_d_jsd[np.logical_and(tmp_category == "healthy", tmp_study_uid == my_study)])
        healthy_avg_dist_bc_in[i] = np.mean(tmp_d_bc[np.logical_and(tmp_category == "healthy", tmp_study_uid == my_study)])
        dysbiosis_avg_dist_jsd_in[i] = np.mean(tmp_d_jsd[np.logical_and(tmp_category == "diseased", tmp_study_uid == my_study)])
        dysbiosis_avg_dist_bc_in[i] = np.mean(tmp_d_bc[np.logical_and(tmp_category == "diseased", tmp_study_uid == my_study)])

    avg_dist_df = pd.DataFrame({'_id' : id_vars,
              'h_avg_dist_jsd' : healthy_avg_dist_jsd,
              'd_avg_dist_jsd' : dysbiosis_avg_dist_jsd,
              'h_avg_dist_jsd_in' : healthy_avg_dist_jsd_in,
              'd_avg_dist_jsd_in' : dysbiosis_avg_dist_jsd_in
#              'h_avg_dist_bc' : healthy_avg_dist_bc,
#              'd_avg_dist_bc' : dysbiosis_avg_dist_bc,
#              'h_avg_dist_bc_in' : healthy_avg_dist_bc_in,
#              'd_avg_dist_bc_in' : dysbiosis_avg_dist_bc_in
              })
    print("3. Dissimilarity...completed")
    
    # 4. Dysbiosis index
    print("4. Dysbiosis index")
    index_df = pd.DataFrame({'_id' : id_vars,
                          'dys_index' : sbm.func.dys_index(mtp_data),
                          'lact_index' : sbm.func.la_index(mtp_data),
                          'scfa_index' : sbm.func.scfa_index(mtp_data),
                          'gms' : sbm.func.gms(mtp_data)
                          })
    print("4. Dysbiosis index...compledted")
    
    # 5. Microbiota abundance
    print("5. Profiling family, class level")
    family_ra = sbm.mtp.util.ra(mtp_data,'family')
    class_ra = sbm.mtp.util.ra(mtp_data,'class')
    
    # 모든 샘플에서 평균 0.01% 안되는 family, class 삭제
    family_pass_ra = family_ra.mean(axis=0,skipna=True) >= 0.0001
    class_pass_ra = class_ra.mean(axis=0,skipna=True) >= 0.0001

    # clr transformation
    all_table_fa_clr = sbm.mtp.util.clr_transform(all_table_fa)
    all_table_cl_clr = sbm.mtp.util.clr_transform(all_table_cl)
    
    all_table_fa_clr = all_table_fa_clr[family_pass_ra.index[family_pass_ra]]
    all_table_cl_clr = all_table_cl_clr[class_pass_ra.index[class_pass_ra]]

    all_table_fa_clr['_id'] = id_vars
    all_table_cl_clr['_id'] = id_vars
    print("5. Profiling family, class level...completed")
    
    # 6. Merge all data
    print("6. Merge data")
    res = mtp_meta[['_id','host_category','host_disease','study_uid','host_age','host_sex','host_bmi','platform','run_number',
                    'host_weight','host_height']]
    print(res.shape)
    res = pd.merge(res,alpha_df,left_on='_id',right_on='_id',how="left")
    print(res.shape)
    res = pd.merge(res,dbrda_coord_jsd,left_on='_id',right_on='_id',how="left")
    print(res.shape)
    #res = pd.merge(res,dbrda_coord_bc,left_on='_id',right_on='_id',how="left")
    #print(res.shape)
    res = pd.merge(res,avg_dist_df,left_on='_id',right_on='_id',how="left")
    print(res.shape)
    res = pd.merge(res,index_df,left_on='_id',right_on='_id',how="left")
    print(res.shape)
    res = pd.merge(res,all_table_fa_clr,left_on='_id',right_on='_id',how="left")
    print(res.shape)
    res = pd.merge(res,all_table_cl_clr,left_on='_id',right_on='_id',how="left")
    print(res.shape)
    print("6. Merge...completed")

    # 7. (add) filtering
    res = res.loc[np.array(total_read_cnt) >= 1000].copy() 
    
    return res


def selecting_features(raw_data, add_gms = True):
    
    # meta, traits data 분리
    sample_meta_features = ['_id','host_category','host_disease','study_uid','host_age','host_sex',
                            'host_bmi','host_weight','host_height','total_read_cnt',
                            'platform','run_number','gms']
    
    sample_meta_table = raw_data[sample_meta_features]
    bio_traits_table = raw_data.drop(sample_meta_features,axis=1)

    # correlation 계산
    cor_table = bio_traits_table.corr(method='spearman')
    cor_table.to_csv("marker_cor_final.csv")

    # r 실행하여 feature selection
    os.system('Rscript r_scripts/feature_selection_using_greedy_selection.R marker_cor_final.csv')
    
    # load output
    selected_marker = pd.read_csv("heuristic_selected_marker.txt",header=None)
    selected_marker = selected_marker[0].to_list()

    # sub traits data table 생성
    selected_traits_table = bio_traits_table[selected_marker]
    selected_traits_table.index = sample_meta_table['_id']
    
    # merge with meta data
    selected_raw_data = pd.merge(sample_meta_table,selected_traits_table,left_on='_id',right_index=True)
    
    # 기존 gms 고려할지 안할지에 대한 option
    if ~add_gms:
        selected_raw_data.drop(['gms'],axis=1)
    
    return selected_raw_data


def make_linear_model_formula(selected_input_data):

    # meta data가 disease state에 주는 효과를 제거하기 위해 linear model의 residual을 이용할 예정인데
    # variable 형태에 따라 random variable을 고려한 모형을 고려해야 함
    # platform, run_number를 고려할 경우 random variable로 취급하기로 정함
    # study 마다 결측이 20% 미만일 경우, 모형 변수로 추가
  
    study_uids = selected_input_data.study_uid.unique()

    form_list = []
    random_list = []

    for stdy in study_uids:
        sub_dat = selected_input_data[selected_input_data['study_uid'] == stdy].copy()
        
        if sub_dat.total_read_cnt.isnull().sum() / sub_dat.shape[0] <= 0.2:
            tmp_formula = "total_read_cnt "
        else :
            tmp_formula = ""
            
        if sub_dat.host_bmi.isnull().sum() / sub_dat.shape[0] <= 0.2:
            tmp_formula = tmp_formula + "+ host_bmi "
        if sub_dat.host_sex.isnull().sum() / sub_dat.shape[0] <= 0.2:
            arr = sub_dat.host_sex.unique()
            idx = [x is not np.nan for x in arr]
            if len(arr[idx]) != 1:
                tmp_formula = tmp_formula + "+ host_sex "
        if sub_dat.host_age.isnull().sum() / sub_dat.shape[0] <= 0.2:
            tmp_formula = tmp_formula + "+ host_age "
        
        if sub_dat.platform.isnull().sum() / sub_dat.shape[0] <= 0.2:
            arr = sub_dat.platform.unique()
            idx = [x is not np.nan for x in arr]
            if len(arr[idx]) != 1:
                tmp_formula = tmp_formula + "+ (1 | platform) "
                isRandom = "T"
            else:
                isRandom = "F"
        else:
            isRandom = "F"

        # run_number는 sample 별로 붙어있는게 많아서 문제다
        if sub_dat.run_number.isnull().sum() / sub_dat.shape[0] <= 0.2:
            arr = sub_dat.run_number.unique()
            idx = [x is not np.nan for x in arr]
            arr2 = sub_dat.run_number
            idx2 = [x is not np.nan for x in arr2]
            if len(arr[idx]) != len(arr2[idx2]):
                tmp_formula = tmp_formula + "+ (1 | run_number) "
                isRandom = "T"
            else:
                isRandom = "F"
        else:
            isRandom = "F"
        
        form_list.append(tmp_formula)
        random_list.append(isRandom)

    pd.DataFrame({
        'study_uid' : study_uids,
        'variables' : form_list,
        'israndom' : random_list
    }).to_csv("lmm_formula.tsv",sep="\t")


def evaluate_marker_traits(selected_input_data_transformed):
    
    ## 1_randomforest_model_to_predict_disease_by_study
    ## check auc by study

    studies = selected_input_data_transformed.study_uid.unique()
    
    param_grid = {
        # Number of trees in random forest
        'n_estimators' : [int(x) for x in np.linspace(start = 20, stop = 200, num = 5)],
        # Number of features to consider at every split
        'max_features' : ['auto', 'sqrt'],
        # Maximum number of levels in tree
        'max_depth' : [int(x) for x in np.linspace(1, 45, num = 3)],
        # Minimum number of samples required to split a node
        'min_samples_split' : [5, 10]
    }
    
    print("1. Randomforest model")
    
    study_uid, diseases, aucs = [], [], []
    
    for stdy in studies:
        stdy_dat = selected_input_data_transformed[selected_input_data_transformed['study_uid'] == stdy]
        tmp_dis = stdy_dat['host_disease'].unique()
        tmp_dis = tmp_dis[~pd.isnull(tmp_dis)]
        disease_list = tmp_dis[~pd.Series(tmp_dis).str.contains("\\|")]
#        disease_list = disease_list[~pd.Series(disease_list).str.contains("ibs_")]

        for dis_nm in disease_list:
            print("Study : " + str(stdy) + ", disease : " + dis_nm)

            # split case, control data in a study
            ctrl_tmp_dat = stdy_dat[stdy_dat['host_category'] == 'healthy']

            tmp_idx = pd.Series(stdy_dat['host_disease']).str.contains(dis_nm)
            tmp_idx[pd.isnull(tmp_idx)] = False
            case_tmp_dat = stdy_dat[tmp_idx]

            # merge speicific disease case dataset and control dataset
            tmp_dat = pd.concat([ctrl_tmp_dat, case_tmp_dat], axis = 0)
            tmp_dat = tmp_dat.drop(['host_disease'],axis=1)
            tmp_dat = tmp_dat.dropna(axis=0)

            # design random forest, k fold cross validation stratified
            seed = 0
            kfold = StratifiedKFold(n_splits=5, shuffle = True, random_state = seed)
            clf = RandomForestClassifier(class_weight = "balanced")

            # dependant, independant variable
            y = pd.get_dummies(tmp_dat.host_category)['diseased']
            X = tmp_dat.drop(['host_category','study_uid','gms'], axis=1)
            
            # grid search to find optimized hyperparameter for random forest by study
            gridforest = GridSearchCV(clf, param_grid, cv = kfold.split(X,y), n_jobs = -1, iid = True)
            gridforest.fit(X, y)

            # calculated auc to measure optimized random forest performance
            y_pred = cross_val_predict(gridforest.best_estimator_,X, y, method = "predict_proba", cv = kfold.split(X,y))[:,1]
            auc_tmp = roc_auc_score(y_true = y.astype(int),
                                    y_score = y_pred)
            study_uid.append(stdy)
            diseases.append(dis_nm)
            aucs.append(np.mean(auc_tmp))
    
    specific_rf_model_res = pd.DataFrame({
        'study_uid' : study_uid,
        'disease' : diseases,
        'AUC' : aucs
    })
    
    print("1. Randomforest model ... completed")
    
    print("2. Calculate marker prediction score")
    
    study_uid, diseases, marker, aucs, signs = [], [], [], [], []

    for stdy in studies:
        stdy_dat = selected_input_data_transformed[selected_input_data_transformed['study_uid'] == stdy]
        tmp_dis = stdy_dat['host_disease'].unique()
        tmp_dis = tmp_dis[~pd.isnull(tmp_dis)]
        disease_list = tmp_dis[~pd.Series(tmp_dis).str.contains("\\|")]
#        disease_list = disease_list[~pd.Series(disease_list).str.contains("ibs_")]

        for dis_nm in disease_list:
            print("Study : " + str(stdy) + ", disease : " + dis_nm)

            # split case, control data in a study
            ctrl_tmp_dat = stdy_dat[stdy_dat['host_category'] == 'healthy']

            tmp_idx = pd.Series(stdy_dat['host_disease']).str.contains(dis_nm)
            tmp_idx[pd.isnull(tmp_idx)] = False
            case_tmp_dat = stdy_dat[tmp_idx]

            # merge speicific disease case dataset and control dataset
            tmp_dat = pd.concat([ctrl_tmp_dat, case_tmp_dat], axis = 0) # axis = 0 (default)
            tmp_dat = tmp_dat.drop(['host_disease'],axis=1)
            tmp_dat = tmp_dat.dropna(axis=0)

            # design random forest, k fold cross validation stratified
            y = pd.get_dummies(tmp_dat.host_category)['diseased']
            X = tmp_dat.drop(['host_category','study_uid'], axis=1)
            markers = X.columns.to_list()

            # triats' performance as marker by study, disease
            for mker in markers:

                roc_auc_score(y, X.loc[:,mker])

                study_uid.append(stdy)
                diseases.append(dis_nm)
                marker.append(mker)
                tmp_auc = roc_auc_score(y, X.loc[:,mker])
                if tmp_auc < 0.5:
                    tmp_auc = 1 - tmp_auc
                    signs.append(-1)
                else:
                    signs.append(1)
                aucs.append(tmp_auc)
                
    marker_prediction_df = pd.DataFrame({
        'study_uid' : study_uid,
        'disease' : diseases,
        'marker_traits' : marker,
        'AUC' : aucs,
        'sign' : signs
    })
    
    print("2. Calculate marker prediction score ... completed")
    
    print("3. Visualization")
    
    specific_rf_model_res['rank'] = 0
    specific_rf_model_res['sign'] = 0
    specific_rf_model_res['marker_traits'] = specific_rf_model_res['disease']
    
    marker_prediction_df['rank'] = marker_prediction_df.groupby(['study_uid','disease'])['AUC'].rank(ascending=False)
    
    viz_dat = pd.concat([marker_prediction_df,specific_rf_model_res],axis=0,sort=False)
    viz_dat['x_label'] = viz_dat['study_uid'].astype(str) + "_" + viz_dat['marker_traits']
    
    viz_dat_top5 = viz_dat[viz_dat['rank'].isin([0,1,2,3,4,5])]
    viz_dat_top5_soted = viz_dat_top5.sort_values(by=['study_uid','disease','rank'])
    
    aucs = viz_dat_top5_soted.AUC
    x_label = viz_dat_top5_soted.x_label

    my_color = np.where(viz_dat_top5_soted.sign == 1, 'orangered', np.where(viz_dat_top5_soted.sign == -1,'royalblue','forestgreen'))
    
    plt.figure(figsize=(6, 30))
    plt.title('Disease specific model AUC, Top5 traits AUCs (> 0.5)')
    plt.barh(range(len(x_label)), aucs[::-1], color=my_color[::-1], align='center')
    plt.yticks(range(len(x_label)), x_label[::-1])
    plt.xlabel('Area under curve')
    plt.xlim([0.5,1])
    plt.savefig('specific_model_top5_traits_AUCs.pdf', bbox_inches = "tight")
    plt.close()
    
    pvt = marker_prediction_df.pivot_table("AUC",index=["sign","marker_traits"])
    pvt['rank'] = pvt.groupby(['sign'])['AUC'].rank(ascending=False)
    
    dominant = list()
    x_label = list()
    for (x, y) in pvt.index.to_list():
        dominant.append('diseased' if x == 1 else 'healthy')
        x_label.append(y)
        
    pvt['x_label'] = x_label
    pvt['dominant'] = dominant
    sub_pvt = pvt[pvt['rank'].isin([range(15)])]
    sorted_sub_pvt = sub_pvt.sort_values(by=['dominant','rank'])
    
    aucs = sorted_sub_pvt.AUC
    x_label = sorted_sub_pvt.x_label

    my_color = np.where(sorted_sub_pvt.dominant == 'diseased', 'orangered', 'royalblue')

    plt.figure(figsize=(6, 30))
    plt.title('Traits Average AUCs (> 0.5)')
    plt.barh(range(len(x_label)), aucs[::-1], color=my_color[::-1], align='center')
    plt.yticks(range(len(x_label)), x_label[::-1])
    plt.xlabel('Area under curve')
    plt.axvline(x=0.5,color='y')
    plt.xlim([0.5,1])
    plt.savefig('Average_marker_AUCs.pdf', bbox_inches = "tight")
    plt.close()
    
    print("3. Visualization ... save OK")
    
    return viz_dat


def find_hyperparameter_kfold(X, y, model_name="lasso"):

    if model_name == "lasso":
        param = {'c_values' : np.logspace(-2, 3, 100)}
        
    elif model_name == "ridge":
        param = {'c_values' : np.logspace(-2, 3, 100)}
    
    elif model_name == "elasticnet":
        param = {'c_values' : np.logspace(-2, 3, 100),
                'lr_values' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        }
        
    elif model_name == "svm":
        param = {'c_values' : [0.1, 1, 10, 50, 100, 500, 1000],
                'gm_values' : [1, 0.5, 0.1, 0.05, 0.01, 0.05, 0.001, 0.0005, 0.0001]
        }
        
    elif model_name == "randomforest":
        param = {# Number of trees in random forest
            'n_estimators' : [int(x) for x in np.linspace(start = 20, stop = 200, num = 5)],
            # Number of features to consider at every split
            'max_features' : ['auto', 'sqrt'],
            # Maximum number of levels in tree
            'max_depth' : [int(x) for x in np.linspace(1, 45, num = 3)],
            # Minimum number of samples required to split a node
            'min_samples_split' : [5, 10]
        }
    
    elif model_name == "gbm":
        param = {
            'n_estimators' : [int(x) for x in np.linspace(start = 20, stop = 200, num = 5)],
            'max_features' : ['auto', 'sqrt'],
            'max_depth' : [int(x) for x in np.linspace(1, 45, num = 3)],
            'min_samples_leaf' : [3, 5, 7, 10],
            'min_samples_split' : [2, 3, 5, 10],
            'learning_rate' : [0.05, 0.1, 0.15, 0.2]
        }
    
    else:
        print("Unknown or incorrect model name")
        return

    # hyperparameter combination
    allNames = param
    combinations = it.product(*(param[Name] for Name in allNames))
    param_list = list(combinations)
    
    # k-fold cross validation
    kfold = StratifiedKFold(n_splits=5, shuffle= True, random_state = 0)
    
    # evaluation auc each case
    res_auc_each_case = []
    
    itr = 0
        
    for param_itr in param_list:
        
        itr = itr + 1
        print(str(itr) + " / " + str(len(param_list)) + " ....ing")
        
        if model_name == "lasso":
            model = LogisticRegression(penalty = 'l1', C = param_itr[0], class_weight = "balanced",solver='liblinear')
        elif model_name == "ridge":
            model = LogisticRegression(penalty = 'l2', C = param_itr[0], class_weight = "balanced",solver='liblinear')
        elif model_name == "elasticnet":
            model = LogisticRegression(penalty = 'elasticnet', C = param_itr[0], l1_ratio = param_itr[1],
                                      class_weight = "balanced", solver='saga')
        elif model_name == "svm":
            model = SVC(C = param_itr[0], gamma = param_itr[1], class_weight = "balanced", probability = True)
        elif model_name == "randomforest":
            model = RandomForestClassifier(n_estimators = param_itr[0],
                                           max_depth = param_itr[2],
                                           min_samples_split = param_itr[3],
                                           max_features = param_itr[1],
                                           n_jobs = -1,
                                           class_weight = "balanced")
        elif model_name == "gbm":
            model = GradientBoostingClassifier(
                n_estimators = param_itr[0],
                max_depth = param_itr[2],
                min_samples_split = param_itr[4],
                max_features = param_itr[1],
                min_samples_leaf = param_itr[3],
                learning_rate = param_itr[5]
            )
            
        else:
            print("Unknown or incorrect model name")
            return
        
        res_tmp = pd.DataFrame(columns=['study_uid','y_true','y_pred'])
    
        for i, (train_idx, test_idx) in enumerate(kfold.split(X, y)):

            # split train, test
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # study weight
            study_weight = 1 / pd.crosstab(X_train['study_uid'],columns="count")
            tmp_df = pd.DataFrame(X_train['study_uid'])
            weight_df = pd.merge(tmp_df,study_weight,left_on="study_uid",right_index=True)

            tmp_study_uid_test = X_test['study_uid']
            X_train = X_train.drop(['study_uid','host_disease'],axis=1)
            X_test = X_test.drop(['study_uid','host_disease'],axis=1)

            # fit model and predict diseased status
            model.fit(X_train,y_train,sample_weight=weight_df['count'])

            y_pred = model.predict_proba(X_test)

            tmp = pd.DataFrame({'study_uid' : tmp_study_uid_test,
                         'y_true' : y_test,
                         'y_pred' : y_pred[:,1],
                })
            res_tmp = pd.concat([res_tmp,tmp])
            
        auc_tmp = []
            
        for s_study in np.unique(res_tmp.study_uid):
            auc_tmp.append(roc_auc_score(y_true = res_tmp[res_tmp.study_uid == s_study].y_true.astype(int),
                                         y_score = res_tmp[res_tmp.study_uid == s_study].y_pred))

        mean_auc = np.mean(auc_tmp)
        res_auc_each_case.append(mean_auc)

    optimized_para = param_list[pd.Series(res_auc_each_case).idxmax()]
    
    if model_name == "lasso":
        optimized_para_dict = {'c_values' : optimized_para[0]}
        
    elif model_name == "ridge":
        optimized_para_dict = {'c_values' : optimized_para[0]}
    
    elif model_name == "elasticnet":
        optimized_para_dict = {'c_values' : optimized_para[0],
                'lr_values' : optimized_para[1]}
        
    elif model_name == "svm":
        optimized_para_dict = {'c_values' : optimized_para[0],
                'gm_values' : optimized_para[1]
        }
        
    elif model_name == "randomforest":
        optimized_para_dict = {
            'n_estimators' : optimized_para[0],
            'max_features' : optimized_para[1],
            'max_depth' : optimized_para[2],
            'min_samples_split' : optimized_para[3]
        }

    elif model_name == "gbm":
        optimized_para_dict = {
            'n_estimators' : optimized_para[0],
            'max_features' : optimized_para[1],
            'max_depth' : optimized_para[2],
            'min_samples_leaf' : optimized_para[3],
            'min_samples_split' : optimized_para[4],
            'learning_rate' : optimized_para[5]
        }
        
    else:
        print("Unknown or incorrect model name")
        return
        
    return optimized_para_dict

def evaluate_best_classifier_model(X, y, model_name, optimized_hparam):
    if model_name == "lasso":
        model = LogisticRegression(penalty = 'l1', C = optimized_hparam['c_values'], class_weight = "balanced",solver='liblinear')
    elif model_name == "ridge":
        model = LogisticRegression(penalty = 'l2', C = optimized_hparam['c_values'], class_weight = "balanced",solver='liblinear')
    elif model_name == "elasticnet":
        model = LogisticRegression(penalty = 'elasticnet', C = optimized_hparam['c_values'],
                                   l1_ratio = optimized_hparam['lr_values'],
                                  class_weight = "balanced", solver='saga')
    elif model_name == "svm":
        model = SVC(C = optimized_hparam['c_values'], gamma = optimized_hparam['gm_values'], class_weight = "balanced", probability = True)
    elif model_name == "randomforest":
        model = RandomForestClassifier(n_estimators = optimized_hparam['n_estimators'],
                                       max_depth = optimized_hparam['max_depth'],
                                       min_samples_split = optimized_hparam['min_samples_split'],
                                       max_features = optimized_hparam['max_features'],
                                       n_jobs = -1,
                                       class_weight = "balanced")
    elif model_name == "logistic":
        model = LogisticRegression(penalty = 'none', class_weight = "balanced", solver='saga')
    elif model_name == "gbm":
            model = GradientBoostingClassifier(
                n_estimators = optimized_hparam['n_estimators'],
                max_depth = optimized_hparam['max_depth'],
                min_samples_split = optimized_hparam['min_samples_split'],
                max_features = optimized_hparam['max_features'],
                min_samples_leaf = optimized_hparam['min_samples_leaf'],
                learning_rate = optimized_hparam['learning_rate']
            )
    else:
        print("Unknown or incorrect model name")
        return
    
    seed = 0
    kfold = StratifiedKFold(n_splits=5, shuffle= True, random_state = seed)

    res_tmp = pd.DataFrame(columns=['study_uid','host_disease','y_true','y_pred'])
    
    for i, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        study_weight = 1 / pd.crosstab(X_train['study_uid'],columns="count")
        tmp_df = pd.DataFrame(X_train['study_uid'])
        weight_df = pd.merge(tmp_df,study_weight,left_on="study_uid",right_index=True)

        tmp_study_uid_test = X_test['study_uid']
        tmp_host_disease_test = X_test['host_disease']
        X_train = X_train.drop(['study_uid','host_disease'],axis=1)
        X_test = X_test.drop(['study_uid','host_disease'],axis=1)

        model.fit(X_train,y_train,sample_weight=weight_df['count'])

        y_pred = model.predict_proba(X_test)

        tmp = pd.DataFrame({'study_uid' : tmp_study_uid_test,
                     'host_disease' : tmp_host_disease_test,
                     'y_true' : y_test,
                     'y_pred' : y_pred[:,1]})
        res_tmp = pd.concat([res_tmp,tmp])


    auc_tmp = []
    n = []
    study_tmp = []
    disease_tmp = []

    for s_study in np.unique(res_tmp.study_uid):
        stdy_dat = res_tmp[res_tmp.study_uid == s_study]
        tmp_dis = stdy_dat['host_disease'].unique()
        tmp_dis = tmp_dis[~pd.isnull(tmp_dis)]
        disease_list = tmp_dis[~pd.Series(tmp_dis).str.contains("\\|")]
#        disease_list = disease_list[~pd.Series(disease_list).str.contains("ibs_")]
        for d_disease in disease_list:
            ctrl_tmp = stdy_dat[stdy_dat.host_disease.isnull()]
            case_tmp = stdy_dat[~stdy_dat.host_disease.isnull()]
            case_tmp_final = case_tmp[case_tmp.host_disease.str.contains(d_disease)]
            if(case_tmp_final.shape[0] == 0):
                continue

            auc_tmp_study_res_tmp = pd.concat([case_tmp_final,ctrl_tmp])
            auc_tmp.append(roc_auc_score(y_true = auc_tmp_study_res_tmp.y_true.astype(int),
                                         y_score = auc_tmp_study_res_tmp.y_pred))
            n.append(auc_tmp_study_res_tmp.shape[0])
            study_tmp.append(s_study)
            disease_tmp.append(d_disease)

    auc_df = pd.DataFrame({
        "model" : model_name,
        "study_uid" : study_tmp,
        "host_disease" : disease_tmp,
        "n_samples" : n,
        "auc" : auc_tmp
    })
    
    results = {"scores" : res_tmp, "auc" : auc_df}
    
    return results
    
def modelling_to_classify_dysbiosis(selected_input_data_transformed, save_model = False, model_name = "randomforest"):
    
    print(model_name + " model diseased v.s. healthy")

    # test set   
    test_set = selected_input_data_transformed[selected_input_data_transformed['study_uid'] == 509]
    diseases = test_set.host_disease
    test_set_X = test_set.drop(['host_category','gms','host_disease'],axis=1)
    test_set_X = test_set_X.drop(['study_uid'],axis=1)
    test_set_y = pd.get_dummies(test_set.host_category)['diseased']
    
    # train set
    dat = selected_input_data_transformed[selected_input_data_transformed['study_uid'] != 509]
    
    dat['study_uid'] = dat['study_uid'].astype(str).copy()
    dat = dat.loc[~dat['gms'].isnull()]
    dat = dat.drop(['gms'],axis=1)
    
    # input (X) 정의, dysbiosis 변수 제거
    X = dat.drop(['host_category'],axis=1)
    
    # 반응변수(y) dummy 변수
    y = pd.get_dummies(dat.host_category)['diseased']
    
    # search optimized hyperparameter
    optimized_hparam = find_hyperparameter_kfold(X, y, model_name)
    model_results = evaluate_best_classifier_model(X, y, model_name, optimized_hparam = optimized_hparam)
    
    # Randomforest_for_gmi_score_by_study (train auc)
    box_fig = sbm.vis.plot.box(model_results['scores']['y_pred'], model_results['scores']['study_uid'], subgroup=model_results['scores']['y_true'], figsize=(12,5))
    sbm.vis.plot.save_fig(box_fig, model_name + '_for_gmi_score_by_study.pdf')
    sbm.vis.plot.close_fig()
    
    if model_name == "lasso":
        model = LogisticRegression(penalty = 'l1', C = optimized_hparam['c_values'], class_weight = "balanced",solver='liblinear')
    elif model_name == "ridge":
        model = LogisticRegression(penalty = 'l2', C = optimized_hparam['c_values'], class_weight = "balanced",solver='liblinear')
    elif model_name == "elasticnet":
        model = LogisticRegression(penalty = 'elasticnet', C = optimized_hparam['c_values'],
                                   l1_ratio = optimized_hparam['lr_values'],
                                  class_weight = "balanced", solver='saga')
    elif model_name == "svm":
        model = SVC(C = optimized_hparam['c_values'], gamma = optimized_hparam['gm_values'], class_weight = "balanced", probability = True)
    elif model_name == "randomforest":
        model = RandomForestClassifier(n_estimators = optimized_hparam['n_estimators'],
                                       max_depth = optimized_hparam['max_depth'],
                                       min_samples_split = optimized_hparam['min_samples_split'],
                                       max_features = optimized_hparam['max_features'],
                                       n_jobs = -1,
                                       class_weight = "balanced")
    elif model_name == "logistic":
        model = LogisticRegression(penalty = 'none', class_weight = "balanced", solver='saga')
    elif model_name == "gbm":
            model = GradientBoostingClassifier(
                n_estimators = optimized_hparam['n_estimators'],
                max_depth = optimized_hparam['max_depth'],
                min_samples_split = optimized_hparam['min_samples_split'],
                max_features = optimized_hparam['max_features'],
                min_samples_leaf = optimized_hparam['min_samples_leaf'],
                learning_rate = optimized_hparam['learning_rate']
            )
    else:
        print("Unknown or incorrect model name")
        return
    
    # predict test set
    study_weight = 1 / pd.crosstab(X['study_uid'],columns="count")
    tmp_df = pd.DataFrame(X['study_uid'])
    weight_df = pd.merge(tmp_df,study_weight,left_on="study_uid",right_index=True)

    X = X.drop(['study_uid','host_disease'],axis=1)

    model.fit(X, y,sample_weight=weight_df['count'])
    test_set_y_pred = model.predict_proba(test_set_X)
    
    test_set_res = pd.DataFrame({
                    'host_disease' : diseases,
                    'y_true' : test_set_y,
                    'y_pred' : test_set_y_pred[:,1],
                })

    tmp_dis = diseases.unique()
    tmp_dis = tmp_dis[~pd.isnull(tmp_dis)]
    disease_list = tmp_dis[~pd.Series(tmp_dis).str.contains("\\|")]
#    disease_list = disease_list[~pd.Series(disease_list).str.contains("ibs_")]
    
    study_tmp = 509
    disease_tmp = []
    n = []
    auc_tmp = []

    for d_disease in disease_list:
        ctrl_tmp = test_set_res[test_set_res.host_disease.isnull()]
        case_tmp = test_set_res[~test_set_res.host_disease.isnull()]
        case_tmp_final = case_tmp[case_tmp.host_disease.str.contains(d_disease)]
        if(case_tmp_final.shape[0] == 0):
            continue

        auc_tmp_study_res_tmp = pd.concat([case_tmp_final,ctrl_tmp])
        auc_tmp.append(roc_auc_score(y_true = auc_tmp_study_res_tmp.y_true.astype(int),
                                        y_score = auc_tmp_study_res_tmp.y_pred))
        n.append(auc_tmp_study_res_tmp.shape[0])
        disease_tmp.append(d_disease)

    auc_df = pd.DataFrame({
        "model" : model_name,
        "study_uid" : study_tmp,
        "host_disease" : disease_tmp,
        "n_samples" : n,
        "auc" : auc_tmp
    })
    
    group_name = np.where(test_set_res.y_true == 1, 'Diseased', 'Healthy')
    test_box_fig = sbm.vis.plot.box(test_set_res.y_pred,group=group_name)
    sbm.vis.plot.save_fig(test_box_fig, 'Test_(Samsung)_set_GMI_score.pdf')
    sbm.vis.plot.close_fig()
    
    print(model_name + " model disased v.s. healthy ... completed")
    
    pd.DataFrame(model.feature_importances_).to_csv("feature_importance_" + model_name + ".csv")

    if save_model:
        joblib.dump(model, 'Final_'+ model_name +'_trained.pkl')

    results = {"train_results" : model_results, "test_results" : auc_df}

    return results

