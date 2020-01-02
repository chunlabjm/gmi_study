args=commandArgs(TRUE)

library(pvclust)
library(gplots)
library(RColorBrewer)
library(proxy)
library(reghelper)

input_path = args[1]
is_test_mode = as.logical(args[2])

selected_input_data_transformed <- read.table(input_path,sep="\t",head=T,fileEncoding = "utf-8",stringsAsFactors = F)

study_itr <- unique(selected_input_data_transformed$study_uid)
disease_itr <- unique(selected_input_data_transformed$host_disease)
disease_itr <- disease_itr[-grep("\\|",disease_itr)]
disease_itr <- disease_itr[disease_itr != '']

process_itr <- 0 # iteration 변수

for(study_nm in study_itr){
  
  process_itr <- process_itr + 1
  
  study_uid <- c()                        # study_uid 저장 벡터
  diseases <- c()                         # study 내 disease 저장 벡터
  microbiota_markers <- c()               # study 내 marker 저장 벡터
  beta_glm <- c()                         # study 내 marker 효과 저장 벡터
  p.values <- c()                         # study 내 marker 효과에 대한 p-value 저장 벡터
  
  # study 마다 진행하기 위해, 데이터 불러오기
  tmp_dat <- subset(selected_input_data_transformed,study_uid == study_nm)
  
  # marker traits dataset
  bio_traits_dat <- subset(tmp_dat,select=-c(X_id,host_category,host_disease,study_uid))
  y <- subset(tmp_dat,select=c(host_category,host_disease))
  
  # marker traits list 정리                                         
  marker_itr <- names(bio_traits_dat)
  
  # 모든 marker traits에서 LMM과 GLM (generalized linear model; 여기에선 logistic regression) 실행
  for(marker_nm in marker_itr){
    
    glm_tmp_dat <- merge(data.frame(residuals = bio_traits_dat[,marker_nm],row.names=rownames(bio_traits_dat)),y,by="row.names",all.x=T)
    
    # disease 별 marker의 효과를 구하기 위한 for 문
    for(disease_nm in disease_itr){
      
      if(!any(grepl(disease_nm, glm_tmp_dat$host_disease))) next # (그럴리 없겠지만) disease 이름이 matching이 되지 않은 skip
      
      # 여러 질병이 있을 수 있는 study가 있을 수 있기에, 특정 disease와 control datset 불러오기
      sub_glm_tmp_dat <- subset(glm_tmp_dat,grepl(disease_nm, host_disease) | host_category == "healthy")
      
      # healthy가 control임을 알려주는 변수 설정
      sub_glm_tmp_dat$host_category <- factor(sub_glm_tmp_dat$host_category,levels=c('healthy','diseased'))
      
      # imbalanced 문제를 해결하기 위해 control, case 군으로 분리
      ctrl_tmp_dat <- subset(sub_glm_tmp_dat,host_category == "healthy")
      case_tmp_dat <- subset(sub_glm_tmp_dat,host_category != "healthy")
      
      # n_control : n_case = 1:2 or 2:1 을 넘어갈 경우 1:2 or 2:1로 모델링 하기 위한 sampling 단계
      if(nrow(ctrl_tmp_dat) / nrow(case_tmp_dat) > 2) {
        
        set.seed(0)
        ctrl_dat_n <- nrow(case_tmp_dat) * 2
        ctrl_dat <- ctrl_tmp_dat[sample(1:nrow(ctrl_tmp_dat),ctrl_dat_n),]
        f_glm_dat <- rbind(ctrl_dat,case_tmp_dat)
        
      } else if(nrow(ctrl_tmp_dat) / nrow(case_tmp_dat) < 0.5) {
        
        set.seed(0)
        case_dat_n <- nrow(ctrl_tmp_dat) * 2
        case_dat <- case_tmp_dat[sample(1:nrow(case_tmp_dat),case_dat_n),]
        f_glm_dat <- rbind(ctrl_tmp_dat,case_dat)
        
      } else {
        f_glm_dat <- rbind(ctrl_tmp_dat,case_tmp_dat)
      }
      
      # GLM 결과를 저장
      glm_res <- glm(host_category ~ residuals, data= f_glm_dat, family="binomial")
      
      tmp_beta <- beta(glm_res)$coefficients[2,1] # standized beta coefficient
      glm_smr <- summary(glm_res)
      
      study_uid <- c(study_uid, study_nm)
      diseases <- c(diseases,disease_nm)
      microbiota_markers <- c(microbiota_markers,marker_nm)
      beta_glm <- c(beta_glm,tmp_beta)
      
      if(is.na(glm_res$coefficients[2])) {
        p.values <- c(p.values,NA)
      } else {
        p.values <- c(p.values,glm_smr$coefficients['residuals','Pr(>|z|)'])
      }
      
      print(paste(study_nm,", ",disease_nm,", ",marker_nm,"...completed",sep=""))
      
    }
    
  }
  
  res <- data.frame(study_uid,diseases,microbiota_markers,beta_glm,p.values)
  
  itr <- 0
  
  # 계산된 p-value를 FDR값으로 변환하기 위한 for문
  for(disease_nm in unique(res$diseases)){
    itr <- itr + 1
    tmp_res <- subset(res,diseases == disease_nm)
    tmp_res$fdr <- p.adjust(tmp_res$p.values,method = 'fdr')
    
    if(itr == 1) f_res <- tmp_res else f_res <- rbind(f_res,tmp_res)
  }
  
  write.table(f_res,"association_result.txt",row.names=F,col.names=(process_itr == 1),quote=F,sep="\t",append=(process_itr != 1))
  
}

f_dat <- read.table("association_result.txt",head=T,sep="\t")

sig_dat <- subset(f_dat,p.values < 0.05 & !study_uid %in% c(509,479))

colnm <- as.character(unique(paste(sig_dat$study_uid,sig_dat$diseases,sep="_")))
rownm <- as.character(unique(sig_dat$microbiota_markers))

res_mat <- matrix(NA,nrow=length(rownm),ncol=length(colnm))

rownames(res_mat) <- rownm
colnames(res_mat) <- colnm

for(i in 1:length(rownm)){
  for(j in 1:length(colnm)){
    tmp_dat <- subset(sig_dat, paste(study_uid,diseases,sep="_") == colnm[j] & microbiota_markers == rownm[i])
    if(nrow(tmp_dat) != 0) res_mat[i,j] <- tmp_dat$beta_glm else res_mat[i,j] <- 0
  }
}

cosine <- function(x) {
  x <- as.matrix(x)
  y <- t(x) %*% x
  res <- 1 - y / (sqrt(diag(y)) %*% t(sqrt(diag(y))))
  res <- as.dist(res)
  attr(res, "method") <- "cosine"
  return(res)
}

res_clust <- pvclust(t(res_mat),method.dist=cosine,method.hclust="average",nboot=10000,quiet=T)
png(width = 1100, height = 1100, filename = "cosine_average_pvclust.png",res = 220)
plot(res_clust,cex=0.3)
pvrect(res_clust,alpha=0.95)
dev.off()

res_clust <- pvclust(t(res_mat),method.dist=cosine,method.hclust="ward.D2",nboot=10000,quiet=T)
png(width = 1100, height = 1100, filename = "cosine_wardD2_pvclust.png",res = 220)
plot(res_clust,cex=0.3)
pvrect(res_clust,alpha=0.95)
dev.off()

my_palette <- colorRampPalette(c("blue", "white", "red"))(n = 199)

col_breaks = c(seq(-1,-0.001,length=100),  # for red
               seq(0,1,length=100))             # for green

par(mar = c(0,0,0,0))

cosine <- function(x) {
  x <- as.matrix(x)
  y <- t(x) %*% x
  res <- 1 - y / (sqrt(diag(y)) %*% t(sqrt(diag(y))))
  res <- as.dist(res)
  attr(res, "method") <- "cosine"
  return(res)
}

png(width = 1100, height = 1100, filename = "association_heatmap_complete.png",res = 220)
heatmap.2(res_mat,
          notecol="black",      # change font color of cell labels to black
          #distfun = function(x) dist(x, method="cosine"),
          hclustfun = function(x) hclust(x, method="complete"),
          density.info="none",  # turns off density plot inside color legend
          trace="none",         # turns off trace lines inside the heat map
          margins =c(10,9),
          col=my_palette,       # use on color palette defined earlier
          dendrogram="row",     # only draw a row dendrogram
          cexRow = 0.3,
          cexCol = 0.3,
          breaks=col_breaks,
          symbreaks=T,
          key = F,
          lhei=c(0.5,7),
          Colv="NA",
          symm=T)
dev.off()
