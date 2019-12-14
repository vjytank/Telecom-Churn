data <- read.csv("C:/Data Science/Logistic Regression/Set 1/Telco-Customer-Churn.csv")
names(data)
telco <- data[-c(1,3:5,17)]
names(telco)

#Checking missing value
sapply(telco, function(x) sum(is.na(x)))
library(gtools)
telco$TotalCharges[is.na(telco$TotalCharges)] <- mean(!is.na(telco$TotalCharges))     
sapply(telco, function(x) sum(is.na(x)))

# labelling data
levels(telco$gender)
table(telco$gender)
telco$gender <- as.numeric(telco$gender,
                          levels = c("Female","Male"),
                          labels = c(1,2))
table(telco$gender)

levels(telco$PhoneService)
table(telco$PhoneService)
telco$PhoneService <- as.numeric(telco$PhoneService,
                                 levels = c("No","Yes"),
                                 labels = c(1,2))
table(telco$PhoneService)

levels(telco$MultipleLines)
table(telco$MultipleLines)
telco$MultipleLines <- as.numeric(telco$MultipleLines,
                                 levels = c("No phone service","Yes","No"),
                                 labels = c(1,2,3))
table(telco$MultipleLines)

levels(telco$InternetService)
table(telco$InternetService)
telco$InternetService <- as.numeric(telco$InternetService,
                                    levels = c("DSL", "Fiber optic", "No"),
                                    labels = c(2,3,1))
table(telco$InternetService)

levels(telco$OnlineSecurity)
table(telco$OnlineSecurity)
telco$OnlineSecurity <- as.numeric(telco$OnlineSecurity,
                                    labels = c(1,2,3))
table(telco$OnlineSecurity)

levels(telco$OnlineBackup)
table(telco$OnlineBackup)
telco$OnlineBackup <- as.numeric(telco$OnlineBackup,
                                   labels = c(1,2,3))
table(telco$DeviceProtection)

levels(telco$DeviceProtection)
table(telco$DeviceProtection)
telco$DeviceProtection <- as.numeric(telco$DeviceProtection,
                                 labels = c(1,2,3))
table(telco$DeviceProtection)

levels(telco$TechSupport)
table(telco$TechSupport)
telco$TechSupport <- as.numeric(telco$TechSupport,
                                     labels = c(1,2,3))
table(telco$TechSupport)

levels(telco$StreamingTV)
table(telco$StreamingTV)
telco$StreamingTV <- as.numeric(telco$StreamingTV,
                                labels = c(1,2,3))
table(telco$StreamingTV)

levels(telco$StreamingMovies)
table(telco$StreamingMovies)
telco$StreamingMovies <- as.numeric(telco$StreamingMovies,
                                labels = c(1,2,3))
table(telco$StreamingMovies)

levels(telco$Contract)
table(telco$Contract)
telco$Contract <- as.numeric(telco$Contract,
                                    labels = c(1,2,3))
table(telco$Contract)

levels(telco$PaymentMethod)
table(telco$PaymentMethod)
telco$PaymentMethod <- as.numeric(telco$PaymentMethod,
                             labels = c(1,2,3,4))
table(telco$PaymentMethod)

levels(telco$Churn)
telco$Churn <- as.factor(ifelse(telco$Churn == "Yes",1,0))
table(telco$Churn)

str(telco)
telco$gender <- as.factor(telco$gender)
telco$tenure <- as.numeric(telco$tenure)
telco$PhoneService <- as.factor(telco$PhoneService)
telco$MultipleLines <- as.factor(telco$MultipleLines)
telco$InternetService <- as.factor(telco$InternetService)
telco$OnlineSecurity <- as.factor(telco$OnlineSecurity)
telco$OnlineBackup <- as.factor(telco$OnlineBackup)
telco$DeviceProtection <- as.factor(telco$DeviceProtection)
telco$TechSupport <- as.factor(telco$TechSupport)
telco$StreamingTV <- as.factor(telco$StreamingTV)
telco$StreamingMovies <- as.factor(telco$StreamingMovies)
telco$Contract <- as.factor(telco$Contract)
telco$PaymentMethod <- as.factor(telco$PaymentMethod)
telco$MonthlyCharges <- as.numeric(telco$MonthlyCharges)
telco$TotalCharges <- as.numeric(telco$TotalCharges)
str(telco)
cor(telco[c(2,14,15)])
#Data Partition
library(caret)
train <- createDataPartition(telco$Churn, p=0.7,list = FALSE)
training <- telco[train,]
testing <- telco[-train,]

#Model Building
library(gtools)
library(e1071)
set.seed(1)
model1 <- glm(Churn~.,family = 'binomial', data = training)
summary(model1)
Acc(model1)
#relevelling with lowest value
model2 <- glm(Churn~relevel(MultipleLines,ref = 2)+relevel(InternetService,ref = 2)+
                +relevel(OnlineSecurity,ref = 2)+
                relevel(OnlineBackup,ref = 2)+relevel(DeviceProtection,ref = 2)+
                relevel(TechSupport,ref = 2)+relevel(StreamingTV,ref = 2)+
                relevel(StreamingMovies,ref = 2)+Contract+gender+PhoneService+
                PaymentMethod+MonthlyCharges+TotalCharges,
              family = binomial, data = training)
summary(model2)
Acc(model2)

model3 <- glm(Churn~relevel(InternetService,ref = 2)+Contract+PhoneService+
                PaymentMethod+MonthlyCharges+TotalCharges,
              family = binomial, data = training)
summary(model3)
Acc(model3)

model4 <- glm(Churn~relevel(InternetService,ref = 2)+Contract+PhoneService+
              +MonthlyCharges+TotalCharges,
              family = binomial, data = training)
summary(model4)
Acc(model4)


##Prediction
training$probs <- predict(model4, training,type = 'response') 
training$predict <- as.factor(ifelse(training$probs > 0.7,1,0))
table(training$Churn,training$predict)                              
confusionMatrix(training$Churn,training$predict)

##ROC curve
library(ROCR)
predtrain <- predict(model4, training,type = 'response') 
ROCRpred <- prediction(predtrain,training$Churn)
ROCRperf <- performance(ROCRpred,"tpr","fpr")
plot(ROCRperf,col="red")


##prediction on testing data
testing$probs <- predict(model4, testing,type = 'response')
testing$predict <- as.factor(ifelse(testing$probs > 0.7,1,0))
table(testing$Churn,testing$predict)                              
confusionMatrix(testing$Churn,testing$predict)







model5 <- trainControl(glm(Churn~relevel(InternetService,ref = 2)+Contract+PhoneService+
                     +MonthlyCharges+TotalCharges,
                   family = binomial, data = training),kernal = 'linear'),method="k-foldcv"

