#################
# Heuristic selection of microbiota marker traits
# spearman correlation > 0.5 adjacency matrix를 생성
# 각 trait의 degree(True의 갯수)를 구함
# 가장 높은 degree를 'final marker?(동점일 때, 랜덤 선택)'로 정함
# network (= adjacency matrix)에서 final marker와 그것과 연결된 nodes (= traits)를 제거한 후
# 위 과정을 반복함 (갯수로 break 할 수 있으나, 우선은 edge가 있는 경우 모두 selection)
# 1 단계에서 adjacency matrix 생성시 -2 step에서 제거했던 nodes는 재투입하여 선정
#################

args=commandArgs(TRUE)

input_path <- args[1]

dat <- read.table(args[1],sep=",",row.names=1,head=T)

mat <- as.matrix(dat)

# upper diagonal matrix 이용
mat[lower.tri(mat)] <- -9
diag(mat) <- -9

rm_nodes <- c()
itr <- 0
rm_nodes_index <- c()
tmp_rm_nodes_index <- NA
tmp_mat <- mat
traits_nm <- rownames(mat)

while(TRUE){
  itr <- itr + 1
  print(itr)
  print(rm_nodes)
  if(itr == 1) tmp_mat <- mat else {
    tmp_mat <- mat[-c(rm_nodes_index,tmp_rm_nodes_index),-c(rm_nodes_index,tmp_rm_nodes_index)]
  }
  adj_mat <- tmp_mat > 0.4
  dgrees <- rowSums(adj_mat)
  print(max(dgrees))
  if(all(dgrees == 0)) break
  
  rm_nodes <- c(rm_nodes,names(which.max(dgrees)))
  rm_nodes_index <- c(rm_nodes_index,which(traits_nm == names(which.max(dgrees))))
  
  tmp_rm_nodes_index <- which(traits_nm %in% names(which(adj_mat[which.max(dgrees),] > 0)))
}

# python에서 읽을 수 있도록 txt로 저장
write.table(unique(rm_nodes),"heuristic_selected_marker.txt",row.names=F,col.names=F,quote=F,sep=",")