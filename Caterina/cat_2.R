library('caret')

call <- read.table("/Users/caterina/Desktop/B4TM/Assignment/Train_call.txt", header=TRUE)
head(call[,1:7])
dim(call)

call_2 <- as.data.frame(t(call[,]))
head(call_2[,0:10])
call_2$Sample <- rownames(call_2)
rownames(call_2) <- NULL
head(call_2[2835])

clinical <- read.table("/Users/caterina/Desktop/B4TM/Assignment/Train_clinical.txt",header = TRUE)
head(clinical)
Sub <- rep(NA, 100)
for (i in 1:100){
  if (clinical$Subgroup[i]=="HER2+") Sub[i] <- 1
  else if (clinical$Subgroup[i]=="HR+") Sub[i] <- 2
  else if (clinical$Subgroup[i]=="Triple Neg") Sub[i] <- 3
}
print(Sub)
clinical_2 <- cbind(clinical,Sub)

final <- merge(clinical_2, call_2, by="Sample")
head(final[,0:7])

########## multinomial models ##########
X <- as.data.frame(final[,4:2837])
head(X[,0:10])
library(VGAM)
vglm1 <- vglm(final$Sub~X$V1, multinomial)
fir <- lm(final$Sub~X[])

########## hierarchical clustering ########## 

reduct <- final[,4:2837]
dm <- dist(reduct, method="manhattan")
head(dm)
dm2 <- dist(reduct, method="euclidean")
head(dm2)

hcc <- hclust(dm, method="complete")
hcc
plot(hcc, labels=final[,3])
rect.hclust(hcc, k=3, border="red")

# NO
hcs <- hclust(dm, method="single")
hcs
plot(hcs, labels=final[,3])
rect.hclust(hcs, k=3, border="red")

# bad 
hca <- hclust(dm, method="average")
hca
plot(hca, labels=final[,3])
rect.hclust(hca, k=3, border="red")

hcw <- hclust(dm, method="ward.D2")
hcw
plot(hcw, labels=final[,3])
rect.hclust(hcw, k=3, border="red")

par(mfrow=c(2,2))
# il valore di y rappresenta il cluster in cui sono collocati # il colore rappresenta la varieta' di grano
plot(cutree(hcc, k=3), col=final[,3], main="Complete linkage") 
plot(cutree(hcs, k=3), col=final[,3], main="Single linkage") 
plot(cutree(hca, k=3), col=final[,3], main="Average linkage") 
plot(cutree(hcw, k=3), col=final[,3], main="Ward method")
par(mfrow=c(1,1))


table(final$Subgroup, cutree(hcc, k=3), dnn=c("Subgroup", "cluster"))
table(final$Subgroup, cutree(hcs, k=3), dnn=c("Subgroup", "cluster"))
table(final$Subgroup, cutree(hca, k=3), dnn=c("Subgroup", "cluster"))
table(final$Subgroup, cutree(hcw, k=3), dnn=c("Subgroup", "cluster"))


########## k-mean ########## 

km3 <- kmeans(reduct, centers=3)
km3
# explained deviance 70.7 % ==> low
pairs(reduct[,0:5], col=km3$cluster) # 3 groups yes
plot(reduct[,1:2], col=km3$cluster, pch=20)
points(km3$centers[,1:2], pch=17, col=c(1,2,3), cex=2)
# comparison with the known subgroup
table(final$Subgroup, km3$cluster, dnn=c("subgroup", "cluster"))
plot(c(1:dim(reduct)[1]),final[,3],col=km3$cluster,
     xlab="index", ylab="subgroup" )


########## PCA ##########
pc <- prcomp(reduct, scale.=T) # con scale.=T ottengo le PC a partire dalle correlazioni
summary(pc)
# screeplot, very steep 
plot(pc, type="l")



