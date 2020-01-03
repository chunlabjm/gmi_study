args <- commandArgs(TRUE)

library(lme4)
library(reghelper)

selected_input_path = args[1]
is_test_mode = as.logical(args[2])

selected_input_data <- read.table(selected_input_path,sep="\t",row.names=1,head=T,fileEncoding = "utf-8",stringsAsFactors = F)

# 사용자 data (test data)일 경우 실행
# bmi missing을 weight, height로 직접 계산해서 imputation
if(is_test_mode){
  for(i in 1:nrow(selected_input_data)){
    if(is.na(selected_input_data$host_bmi[i])) {
      if(!is.na(selected_input_data$host_weight[i]) & !is.na(selected_input_data$host_height[i])) selected_input_data$host_bmi[i] <- selected_input_data$host_weight[i] / (selected_input_data$host_height[i] / 100)^2
    }
  }
  selected_input_data$host_sex[selected_input_data$host_sex == ''] <- NA
  selected_input_data$host_sex[selected_input_data$host_sex == "0"] <- "male"
  selected_input_data$host_sex[selected_input_data$host_sex == "1"] <- "female"

  # host_disease '|' 로 복수로 있는 경우가 있음 (사용자 데이터)
  disease_itr <- unique(selected_input_data$host_disease)
  disease_itr <- disease_itr[-grep("\\|",disease_itr)]
  disease_itr <- disease_itr[disease_itr != '']
} else {
  disease_itr <- unique(selected_input_data$host_disease)
}

# character -> factor
selected_input_data$host_category <- factor(selected_input_data$host_category,levels = c('diseased','healthy'))
selected_input_data$study_uid <- factor(selected_input_data$study_uid)
selected_input_data$host_sex <- factor(selected_input_data$host_sex)
selected_input_data$host_disease <- factor(selected_input_data$host_disease)
selected_input_data$platform <- factor(selected_input_data$platform)

# study iteration object
study_itr <- unique(selected_input_data$study_uid)

# model method by study
lmm_design <- read.table("test/lmm_design.txt",head = T, sep="\t", stringsAsFactors = F)

process_itr <- 0

res_dat <- data.frame()

for(study_nm in study_itr){
  
  process_itr <- process_itr + 1
  
  print(paste(process_itr," / ",length(study_itr)," ing... study_uid : ",study_nm, sep=""))
  
  study_uid <- c()                        # study_uid 저장 벡터
  diseases <- c()                         # study 내 disease 저장 벡터
  microbiota_markers <- c()               # study 내 marker 저장 벡터
  beta_glm <- c()                         # study 내 marker 효과 저장 벡터
  p.values <- c()                         # study 내 marker 효과에 대한 p-value 저장 벡터
  
  tmp_dat <- subset(selected_input_data,study_uid == study_nm)
  
  # LM (linear model)을 위한 covariate, dependant variable 분리
  covariate_vars <- c('host_age','host_bmi','total_read_cnt','host_sex','host_weight','host_height','host_category','host_disease','run_number','platform')
  covariate_dat <- subset(tmp_dat,select = covariate_vars)

  # marker traits dataset
  bio_traits_dat <- subset(tmp_dat,select=-c(X_id,host_age,host_bmi,total_read_cnt,platform,host_weight,host_height,
                                             host_sex,run_number,host_category,host_disease,study_uid))
  
  # scaling marker traits variables
  for(i in 1:ncol(bio_traits_dat)){
    bio_traits_dat[,i] <- scale(bio_traits_dat[,i])
  }
  
  # (그럴리는 없겠지만) 모든 샘플에서 0 인 marker traits (scaling하면 NaN 되는 marker 제외)
  bio_traits_dat <- bio_traits_dat[,!apply(bio_traits_dat,2,function(x) any(is.nan(x)))]
  
  # marker traits list 정리                                         
  marker_itr <- names(bio_traits_dat)
  
  id_dat <- subset(tmp_dat, select="X_id")
  res_tmp_dat <- subset(tmp_dat, select="X_id")
  
  inner_itr <- 0
  
  # 모든 marker traits에서 LMM과 GLM (generalized linear model; 여기에선 logistic regression) 실행
  for(marker_nm in marker_itr){
    
    inner_itr <- inner_itr + 1
    
    # covariate의 효과 (고정효과, 랜덤효과)를 제거한 marker의 residual을 생성
    test_marker_dat <- subset(bio_traits_dat, select=marker_nm)
    lmm_tmp_dat <- cbind(covariate_dat,test_marker_dat)
    lmm_formula <- as.formula(paste(marker_nm," ~ 1 + ",lmm_design$variables[lmm_design$study_uid == study_nm],sep=""))
    
    if(lmm_design$israndom[lmm_design$study_uid == study_nm]) {
      model <- lmer(lmm_formula,data=lmm_tmp_dat)
    } else {
      model <- lm(lmm_formula,data=lmm_tmp_dat)
    }
    
    mk_dat <- data.frame(marker_nm = resid(model))
    names(mk_dat) <- marker_nm
    id_tmp_dat <- subset(merge(mk_dat,id_dat,by="row.names",all.x=T),select = -Row.names)
    res_tmp_dat <- merge(res_tmp_dat,id_tmp_dat,by="X_id",all.x=T)
    
  }
  
  res_dat <- rbind(res_dat,res_tmp_dat)
  
}

selected_input_data_transformed <- merge(res_dat,selected_input_data[,c("X_id","host_category","host_disease","study_uid")],
                                         by="X_id",all.x=T)

write.table(selected_input_data_transformed,"selected_input_data_transformed.tsv",row.names=F,col.names=T,quote=F,sep="\t")