clinical <- read.table("/Users/caterina/Desktop/B4TM/Assignment/Train_clinical.txt",header = TRUE)
attach(clinical)
head(clinical)
table(Subgroup)

call <- read.table("/Users/caterina/Desktop/B4TM/Assignment/Train_call.txt", header=TRUE)
head(call)
dim(call)
attach(call)
call_reduct <- call[1:4]
head(call_reduct)
summary(call_reduct)
# for each chromosome, how many areas are in loss, gain, normal, amplification
table(Chromosome, Array.129)
table(Array.129,Start)

# nclones: number of strands where the measurements are performed
# so for each sequence is reported the amount of spots in the array
# they reduce dimensionality by determining sequences of clones which for 
# every sample are (almost) constant within the sample
# for a big number of clones, there is a bigger amount of probes


hist(table(Nclone[Chromosome==1]), nclass=50)
# ==> that is decided from the researcher, right?

# analysis for array.129
table(Chromosome, Array.129)
freq <- prop.table(table(Chromosome, Array.129),1)
print(freq)
cumtab <- apply(freq,1,cumsum)
print(cumtab)
barplot(t(freq),beside=T, legend = T, col=2:5, main="Conditioned frequencies")
plot(1:23,cumtab[1,], ylim = c(-1,3),type="b", pch=20, xlab="degree",ylab="log")
points(1:23,cumtab[2,], type="b",  pch=20, col=2)
points(1:23,cumtab[3,], type="b",  pch=20, col=3)
legend("topleft",lty=1,col=1:3, legend=c("-1","0", "1"), bty="n")



## find a summary measure for each array.*:
# for each array I save the relative frequency of the amount of regions that show
# loss, that show gain, that are normal and that show amplification

plot(as.factor(Array.10), Chromosome)
plot(Array.10, Chromosome)
plot(c(Array.10, Array.100, Array.101), Chromosome)
cor(call[5:104])

##### dataset proportional #####
# some arrays don't have 2
for (i in 5:104) {print(prop.table(table(call[i])))}

A <- matrix(NA, 100, 4)
for (i in 5:104)
{
  print(i)
  p <- prop.table(table(call[i]))
  if (length(names(p))==4){
    A[i-4,] <- prop.table(table(call[i]))
  }
  else if (length(names(p))==3) {
    A[i-4,] <- c(prop.table(table(call[i])),0)
  }
}
print(A)
Sub <- rep(NA, 100)
for (i in 1:100){
  if (clinical$Subgroup[i]=="HER2+") Sub[i] <- 1
  else if (clinical$Subgroup[i]=="HR+") Sub[i] <- 2
  else if (clinical$Subgroup[i]=="Triple Neg") Sub[i] <- 3
}
print(Sub)

new_clinical <- cbind(clinical, A, Sub)
#colnames(new_clinical) <- c("Sample", "Subgroup", "-1", "0", "1", "2")
print(new_clinical)
write.table(new_clinical, "/Users/caterina/Desktop/B4TM/Assignment/clinical_rel.txt", col.names = c("Sample", "Subgroup", "loss", "normal", "gain", "amp", "Sub"))


##### dataset absolute #####
A <- matrix(NA, 100, 4)
for (i in 5:104)
{
  print(i)
  p <- table(call[i])
  if (length(names(p))==4){
    A[i-4,] <- table(call[i])
  }
  else if (length(names(p))==3) {
    A[i-4,] <- c(table(call[i]),0)
  }
}
print(A)

new_clinical <- cbind(clinical, A, Sub)
#colnames(new_clinical) <- c("Sample", "Subgroup", "-1", "0", "1", "2")
print(new_clinical)
write.table(new_clinical, "/Users/caterina/Desktop/B4TM/Assignment/clinical_abs.txt", col.names = c("Sample", "Subgroup", "loss", "normal", "gain", "amp", "Sub"))



##### PCA #####





detach(clinical)
detach(call)
rm(list=ls())









