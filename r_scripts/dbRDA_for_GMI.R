args=commandArgs(TRUE)

library(vegan)

jsd_dist_path <- args[1]
bcd_dist_path <- args[2]
condition_path <- args[3]

d_jsd <- read.table(jsd_dist_path,sep=",",head=T,row.names=1)
d_bc <- read.table(bcd_dist_path,sep=",",head=T,row.names=1)
cond <- read.table(condition_path,sep=",")
colnames(cond) <- c('id','cond')
env_var <- cond

d_jsd <- as.dist(d_jsd)
d_bc <- as.dist(d_bc)

db_rda_jsd <- capscale(d_jsd~Condition(env_var$cond))
db_rda_bc <- capscale(d_bc~Condition(env_var$cond))

cat("dbRDA starts \n")
dbrda_coord_jsd = as.data.frame(summary(db_rda_jsd)$sites)
dbrda_coord_bc = as.data.frame(summary(db_rda_bc)$sites)

names(dbrda_coord_jsd) <- paste(names(dbrda_coord_jsd),'_jsd',sep='')
names(dbrda_coord_bc) <- paste(names(dbrda_coord_bc),'_bc',sep='')

write.table(dbrda_coord_jsd,"dbrda_coord_jsd.csv",col.names=T,row.names=T,quote=F,sep=",")
write.table(dbrda_coord_bc,"dbrda_coord_bc.csv",col.names=T,row.names=T,quote=F,sep=",")