### merged final dataset
library(tidyverse)
library(rstudioapi)
library(ggfortify)
library(caret)

setwd(dirname(getActiveDocumentContext()$path))
data <- read.csv2("merged.csv", header = T, sep=";", row.names = 1)
head(data[,1:7])
dim(data)
summary(data[,1:7])

########## multinomial models ##########
library(VGAM)
vglm1 <- vglm(Subgroup~.,data = data, multinomial)
summary(vglm1)

########## feature selection ##########
feature_selection <- function(data, X, y, perc = 0.6){
  fs_matrix <- filterVarImp(X, y)
  fs <- apply(fs_matrix,1,mean)
  sorted_features <- sort(fs, decreasing = T)
  n <- length(sorted_features[sorted_features >= perc])
  selected <- sorted_features[1:n]
  
  keep <- c(names(data[,1:2]), names(selected))
  data_fs <- data[, (names(data) %in% keep)]
  return(data_fs)
}

data_fs <- feature_selection(data, data[,3:2836], as.factor(data[,2]), perc = 0.6)

pc <- prcomp(data_fs[,3:length(data_fs)], scale.=T, center = TRUE) # con scale.=T ottengo le PC a partire dalle correlazioni
summary(pc)
biplot(pc)

df_out <- as.data.frame(pc$x)
df_out$y <- data_fs[,2]
p<-ggplot(df_out,aes(x=PC1,y=PC2,color=y ))
p<-p+geom_point()
p

########## classifier ##########

#### knn #### 
data_feature <- feature_selection(data, data[,3:2836], as.factor(data[,2]), perc = 1)
model_knn <- train(data_feature[,3:length(data_feature)], data_feature[,2], method = "knn",
                   trControl = trainControl(method="cv", number = 10))
model_knn

model_knn$results[model_knn$results[,1]==model_knn$bestTune[1,1]]
max(model_knn$results[,2])
which(max(model_knn$results[,2]))
perc <- seq(0.5, 0.75, by=0.01)
maxx <- kk <- array(NA, length(perc))
j <- 1
for (i in perc) {
  data_feature <- feature_selection(data, data[,3:2836], as.factor(data[,2]), perc = i)
  model_knn <- train(data_feature[,3:length(data_feature)], data_feature[,2], method = "knn",
                     trControl = trainControl(method="cv", number = 10))
  maxx[j] <- max(model_knn$results[,2])
  kk[j] <- model_knn$bestTune[1,1]
  j <- j + 1
  print(i)
}
plot(perc, maxx, type="b")

#### random forest #### 
data_feature <- feature_selection(data, data[,3:2836], as.factor(data[,2]), perc = 0.6)
model_rf <- train(data_feature[,3:length(data_feature)], data_feature[,2], method = "rf",
                   trControl = trainControl(method="repeatedcv", number = 5, repeats = 5))
model_rf

perc <- seq(0.5, 0.75, by=0.05)
maxx <- kk <- array(NA, length(perc))
j <- 1

for (i in perc) {
  data_feature <- feature_selection(data, data[,3:2836], as.factor(data[,2]), perc = i)
  model_rf <- train(data_feature[,3:length(data_feature)], data_feature[,2], method = "rf",
                     trControl = trainControl(method="cv", number = 10))
  maxx[j] <- max(model_rf$results[,2])
  kk[j] <- model_rf$bestTune[1,1]
  j <- j + 1
  print(i)
}
plot(perc, maxx, type="b")
maxx
kk

# perc = 0.65
data_feature <- feature_selection(data, data[,3:2836], as.factor(data[,2]), perc = 0.65)
model_rf <- train(data_feature[,3:length(data_feature)], data_feature[,2], method = "extraTrees",
                  trControl = trainControl(method="cv", number = 10))
model_rf$resample

predict(model_rf)
table(predict(model_rf), data[,2])


##### ranger ####
set.seed(123)
indxTrain <- createDataPartition(y = data[,1], p = 0.75, list = FALSE)
training <- data[indxTrain,]
testing <- data[-indxTrain,]
prop.table(table(training$Subgroup))
prop.table(table(testing$Subgroup))
prop.table(table(data$Subgroup))

tuneGrid <- data.frame(
  .mtry = seq(10, dim(training)[2], by = 200),
  .splitrule = "gini",
  .min.node.size = 5
)
model <- train(
  Subgroup~.,
  tuneGrid = tuneGrid,
  data = training, 
  method = "ranger",
  trControl = trainControl(
    method = "repeatedcv", 
    number = 3,
    repeats = 20,
    verboseIter = TRUE
  )
)
model
plot(model)

pred <- predict(model, newdata = testing)
confusionMatrix(pred, as.factor(testing$Subgroup))

library(pROC)
pred_prob <- predict(model, newdata = testing, type="response")
rfROC <- roc(response = as.factor(testing$Subgroup), predictor = pred_prob, levels = c("HER2+", "HR+"))
rfROC

plot(rfROC, type="S", print.thres= 0.5)


##### rf specific ##### 
library("randomForest")
set.seed(1234)





