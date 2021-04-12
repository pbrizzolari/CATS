##### new dataset
clinical2 <- read.table("/Users/caterina/Desktop/B4TM/Assignment/clinical_rel.txt",header = TRUE)
clinical2 <- read.table("/Users/caterina/Desktop/B4TM/Assignment/clinical_abs.txt",header = TRUE)
head(clinical2)
attach(clinical2)
plot(clinical2[,3:6])
pairs(clinical2[,3:6])

########## EDA ########## 
par(mfrow=c(2,2))
for (i in 3:6) {
  boxplot(clinical2[,i]~Sub)
}
par(mfrow=c(1,1))


########## k-mean ########## 
km1 <- kmeans(clinical2[,3:6], centers=1)

km2 <- kmeans(clinical2[,3:6], centers=2)
km2
pairs(clinical2[,3:6], col=km2$cluster) # 2 groups well defined
plot(clinical2[,5:6], col=km2$cluster, pch=20)
points(km2$centers[,3:4], pch=17, col=c(1,2), cex=2)

km3 <- kmeans(clinical2[,3:6], centers=3)
km3
# explained deviance 70.7 % ==> low
pairs(clinical2[,3:6], col=km3$cluster) # 3 groups yes
plot(clinical2[,5:6], col=km3$cluster, pch=20)
points(km3$centers[,3:4], pch=17, col=c(1,2,3), cex=2)
# comparison with the known subgroup
table(clinical2$Subgroup, km3$cluster, dnn=c("subgroup", "cluster"))
plot(c(1:dim(clinical2)[1]),clinical2[,7],col=km3$cluster,
     xlab="index", ylab="subgroup" )

km4 <- kmeans(clinical2[,3:6], centers=4)
km4
# explained deviance 78.6 % ==> acceptable
# 4 groups well defined
pairs(clinical2[,3:6], col=km4$cluster)
plot(clinical2[,5:6], col=km4$cluster, pch=20)
points(km4$centers[,3:4], pch=17, col=1:4, cex=2)
# comparison with the known subgroup
table(clinical2$Subgroup, km4$cluster, dnn=c("subgroup", "cluster"))
plot(c(1:dim(clinical2)[1]),clinical2[,7],col=km4$cluster,
     xlab="index", ylab="subgroup" )


km5 <- kmeans(clinical2[,3:6], centers=5)
km5
# explained deviance 82.3 % ==> good
pairs(clinical2[,3:6], col=km5$cluster)
plot(clinical2[,5:6], col=km5$cluster, pch=20)
points(km5$centers[,3:4], pch=17, col=1:5, cex=2)
# comparison with the known subgroup
table(clinical2$Subgroup, km5$cluster, dnn=c("subgroup", "cluster"))
plot(c(1:dim(clinical2)[1]),clinical2[,7],col=km5$cluster,
     xlab="index", ylab="subgroup" )

# choosing with variance
km6 <- kmeans(clinical2[,3:6], centers=6)
km7 <- kmeans(clinical2[,3:6], centers=7)
km8 <- kmeans(clinical2[,3:6], centers=8)
explained.var <- c(km1$betweenss/km1$totss, km2$betweenss/km2$totss,
                   km3$betweenss/km3$totss, km4$betweenss/km4$totss,
                   km5$betweenss/km5$totss, km6$betweenss/km6$totss,
                   km7$betweenss/km7$totss, km8$betweenss/km8$totss)
plot(c(1:8), explained.var, xlab="N. clusters",
     ylab="% explained variance", type="o")
# ==> 4 or 5 clusters


########## distance measures ########## 

dm <- dist(clinical2[,3:6], method="manhattan")
head(dm)
dm2 <- dist(clinical2[,3:6], method="euclidean")
head(dm2)

hcc <- hclust(dm, method="complete")
hcc
plot(hcc, labels=clinical2[,7])
rect.hclust(hcc, k=3, border="red")

# NO
hcs <- hclust(dm, method="single")
hcs
plot(hcs, labels=clinical2[,7])
rect.hclust(hcs, k=3, border="red")

hca <- hclust(dm, method="average")
hca
plot(hca, labels=clinical2[,7])
rect.hclust(hca, k=3, border="red")

hcw <- hclust(dm, method="ward.D2")
hcw
plot(hcw, labels=clinical2[,7])
rect.hclust(hcw, k=3, border="red")

par(mfrow=c(2,2))
# il valore di y rappresenta il cluster in cui sono collocati # il colore rappresenta la varieta' di grano
plot(cutree(hcc, k=3), col=clinical2[,7], main="Complete linkage") 
plot(cutree(hcs, k=3), col=clinical2[,7], main="Single linkage") 
plot(cutree(hca, k=3), col=clinical2[,7], main="Average linkage") 
plot(cutree(hcw, k=3), col=clinical2[,7], main="Ward method")
par(mfrow=c(1,1))


table(clinical2$Sub, cutree(hcc, k=3), dnn=c("Subgroup", "cluster"))
table(clinical2$Sub, cutree(hcs, k=3), dnn=c("Subgroup", "cluster"))
table(clinical2$Sub, cutree(hca, k=3), dnn=c("Subgroup", "cluster"))
table(clinical2$Sub, cutree(hcw, k=3), dnn=c("Subgroup", "cluster"))

# it's not working because the group cannot be classified correctly not with the 
# k-mean method or the hierarchical clustering


########## PCA ##########
X <- clinical2[,3:6]
pc <- prcomp(X, scale.=T) # con scale.=T ottengo le PC a partire dalle correlazioni
summary(pc)
# screeplot, very steep 
plot(pc, type="l")

biplot(pc)



detach(clinical2)
rm(list=ls())
