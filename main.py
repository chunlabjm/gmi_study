import os
import sys

import sbm
import numpy as np
import pandas as pd
import gmi_model as gm

## R library
# library(lme4)
# library(reghelper)
# library(vegan)
# library(pvclust)
# library(gplots)
# library(RColorBrewer)
# library(proxy)
# library(reghelper)

print("0_Loading_profile")

# 정형화된 meta data가 없기 때문에 아직은 manual로 진행해야 함

# essensial meta data
# 'host_category','host_disease','study_uid','host_age','host_sex','host_bmi','platform','run_number' 결측 확인
# 'host_category' : diseased v.s. healthy
# 'host_sex' : '' 있는지 확인, 0, 1로 코딩되어 있다면 male, female로 변환
# 'host_bmi' : height와 weight로 계산할 수 있으면 결측값 채우기

#mtp_data = sbm.mtp.load_data('test/test.json')
#mtp_meta = pd.read_csv('test/test_meta.tsv',index_col=0,low_memory=False)

# 0479 : elderly study 제거
#sub_mtp_data = sbm.mtp.filter_data(mtp_data, mtp_meta.study_uid != 479)
#sub_mtp_meta = mtp_meta[mtp_meta.study_uid != 479]

#input_data = gm.generate_bio_traits(sub_mtp_data, sub_mtp_meta)

#selected_input_data = gm.selecting_features(input_data, add_gms=True)
#selected_input_data.to_csv("selected_input_data.tsv",sep='\t')

#gm.make_linear_model_formula(selected_input_data)

#os.system('Rscript r_scripts/calculate_residuals_using_lm.R selected_input_data.tsv TRUE')

selected_input_data_transformed = pd.read_csv("selected_input_data_transformed.tsv",sep="\t",index_col=0)

print("1_Association_study")
os.system('Rscript r_scripts/explore_association_among_traits.R selected_input_data_transformed.tsv TRUE')

print("2_Evaluating_marker_as_predictor")
result_trait_auc = gm.evaluate_marker_traits(selected_input_data_transformed)

print("3_GMI_model")
model_results = gm.modelling_to_classify_dysbiosis(selected_input_data_transformed, save_model = True, model_name = "randomforest")
train_results = model_results['train_results']['auc'].drop(['model'],axis=1)
train_results['study_uid'] = train_results['study_uid'].apply(int)

test_results = model_results['test_results'].drop(['model'],axis=1)
test_results['study_uid'] = test_results['study_uid'].apply(int)

model_results = pd.concat([train_results,test_results], axis=0)

print("4_save_1+2+3_results")
sub_trait_auc = result_trait_auc[result_trait_auc['rank'] == 0]
trait_auc_disease = sub_trait_auc[['study_uid','disease','AUC']]

mker_nm = result_trait_auc[result_trait_auc['rank'] != 0].marker_traits.unique()
for mker in mker_nm:
    tmp = result_trait_auc[result_trait_auc['marker_traits'] == mker][['study_uid','disease','AUC']]
    tmp = tmp.rename(columns={'AUC':mker})
    trait_auc_disease = pd.merge(trait_auc_disease,tmp,on=['study_uid','disease'])
    
trait_auc_disease.rename(columns={'AUC':'AUC_specific'}, inplace=True)

results = pd.merge(trait_auc_disease,model_results[['study_uid','host_disease','auc']],
         left_on=["study_uid","disease"],right_on=["study_uid","host_disease"],how="left").drop(["host_disease"],axis=1)

cols = results.columns.tolist()
#new_col = []
#new_col = new_col + ['study_uid', 'disease', 'auc', 'AUC_specific', 'gms', 'otus', 'shannon', 'd_avg_dist_jsd_in', 'h_avg_dist_jsd_in']
essential_var = ['study_uid', 'disease', 'auc', 'AUC_specific', 'gms', 'otus', 'shannon', 'd_avg_dist_jsd_in', 'h_avg_dist_jsd_in']
new_col = [ev for ev in essential_var if ev in cols]
#new_col = new_col + ['study_uid', 'disease', 'auc', 'AUC_specific', 'gms', 'otus', 'shannon', 'h_avg_dist_jsd_in']
new_col2 = [x for x in cols if not any(y in x for y in new_col)]
new_col = new_col + new_col2

results_ordered = results[new_col].copy()
results_ordered.rename(columns={'auc':'AUC_GMI'}, inplace=True)

results_ordered.to_csv("results_190103_randomforest_BC_removed.csv")

print("4_save_1+2+3_results ... completed")