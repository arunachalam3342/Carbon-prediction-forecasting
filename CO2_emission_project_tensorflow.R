library(recipes)
data<-read.csv('CO2_Emissions.csv')
print(dim(data))
#check missing data
library(dplyr)
lapply(data,function(x) sum(is.na(x))) %>% str()
names(data)
lapply(data,function(x) length(unique(x))) %>% str()
lapply(data,function(x) unique(x)) %>% str()
lapply(data,function(x) class(x)) %>% str()
data$Fuel.Type=factor(data$Fuel.Type,labels = c(1,2,3,4,5))
data$Fuel.Type=as.numeric(data$Fuel.Type)
unique(data$Fuel.Type)
dataset<-recipe(CO2.Emissions.g.km. ~ .,data) %>% 
  step_num2factor(Fuel.Type,levels=c("Z","D","X","E","N")) %>%
  step_dummy(Fuel.Type,one_hot=TRUE) %>%
  prep() %>%
  bake(new_data= NULL)

dataset$Make=as.numeric(factor(dataset$Make,labels=c(1:length(unique(dataset$Make)))))
dataset$Model=as.numeric(factor(dataset$Model,labels=c(1:length(unique(dataset$Model)))))
dataset$Vehicle.Class=as.numeric(factor(dataset$Vehicle.Class,labels=c(1:length(unique(dataset$Vehicle.Class)))))
dataset$Transmission=as.numeric(factor(dataset$Transmission,labels=c(1:length(unique(dataset$Transmission)))))

#split the data
library(tidymodels)
split<-initial_split(dataset,0.8)
train_dataset<-training(split)
test_dataset<-testing(split)

library(tidyverse)
train_dataset %>% dplyr::select(CO2.Emissions.g.km.,Vehicle.Class,Engine.Size.L.,Cylinders,Transmission) %>%
  GGally::ggpairs()

#split features from labels
train_features<-train_dataset %>% dplyr::select(-CO2.Emissions.g.km.)
test_features<-test_dataset %>% dplyr::select(-CO2.Emissions.g.km.)
train_labels<-train_dataset %>% dplyr::select(CO2.Emissions.g.km.)
test_labels<-test_dataset %>% dplyr::select(CO2.Emissions.g.km.)

#Normalization
library(skimr)
library(tensorflow)
library(keras)
skim<-skimr::skim_with(numeric=skimr::sfl(mean,sd))
skimr::skim(train_dataset)
train_dataset %>% dplyr::select(where(~is.numeric(.x))) %>% pivot_longer(cols=everything(),names_to="variable",values_to="values") %>% group_by(variable) %>% summarise(mean=mean(values),sd=sd(values))
normalizer<-layer_normalization(axis=-1L)
normalizer %>% adapt(as.matrix(train_features))
print(normalizer$mean)
first<-as.matrix(train_features[1,])
cat('First Example: ',first)
cat('Normalized:',as.matrix(normalizer(first)))
#Linear regression of single variable of fuels miles per gallon with c02 emission
mpg<-matrix(train_features$Fuel.Consumption.Comb..mpg.)
mpg_normalizer<-layer_normalization(input_shape=shape(1),axis=NULL)
mpg_normalizer %>% adapt(mpg)

#keras sequential model
mpg_model<-keras_model_sequential() %>% mpg_normalizer() %>% layer_dense(units = 1)
summary(mpg_model)
mpg_model %>% compile(optimizer= optimizer_adam(learning_rate = 0.1),loss= 'mean_absolute_error')
history<-mpg_model %>% fit(mpg,as.matrix(train_labels),epochs=100,verbose=0,validation_split=0.2)
plot(history)
test_results<-list()
test_results[['mpg_model']]<-mpg_model %>% evaluate(as.matrix(test_features$Fuel.Consumption.Comb..mpg.),as.matrix(test_labels),verbose=0)
y<-predict(mpg_model,test_features$Fuel.Consumption.Comb..mpg.)
ggplot(train_dataset) + geom_point(aes(x=Fuel.Consumption.Comb..mpg.,y=CO2.Emissions.g.km.,color="data")) + 
  geom_line(data=data.frame(test_features$Fuel.Consumption.Comb..mpg.,y),aes(x=test_features$Fuel.Consumption.Comb..mpg.,y=y,color="prediction"))

#Linear regression with multiple inputs
linear_model <-keras_model_sequential() %>% normalizer() %>% layer_dense(units=1)
summary(linear_model)
linear_model %>% compile(optimizer= optimizer_adam(learning_rate = 0.1),loss='mean_absolute_error')
history<-linear_model %>% fit(as.matrix(train_features),as.matrix(train_labels),epochs=100,verbose=0,validation_split=0.2)
plot(history)
test_results[['linear_model']]<-linear_model %>% evaluate(as.matrix(test_features),as.matrix(test_labels),verbose=0)
y<-predict(linear_model,as.matrix(test_features))

#Regression with DNN
build_and_complie_model <-function(norm){
  model<-keras_model_sequential() %>% norm() %>% 
    layer_dense(64, activation='relu') %>%
    layer_dense(64,activation= "relu") %>%
    layer_dense(1)
  model %>% compile(
    loss='mean_absolute_error',
    optimizer = optimizer_adam(0.001)
  )
  model
}
dnn_mpg_model<-build_and_complie_model(mpg_normalizer)
summary(dnn_mpg_model)
history<-dnn_mpg_model %>% fit(as.matrix(train_features$Fuel.Consumption.Comb..mpg.),as.matrix(train_labels),verbose=0,epochs=100,validation_split=0.2)
plot(history)
test_results[['dnn_mpg_model']]<-dnn_mpg_model %>% evaluate(as.matrix(test_features$Fuel.Consumption.Comb..mpg.),as.matrix(test_labels),verbose=0)
y<-predict(dnn_mpg_model,as.matrix(test_features$Fuel.Consumption.Comb..mpg.))
ggplot(train_dataset) + geom_point(aes(x=Fuel.Consumption.Comb..mpg.,y=CO2.Emissions.g.km.,color="data")) +
  geom_line(data=data.frame(test_features$Fuel.Consumption.Comb..mpg.,y),aes(x=test_features$Fuel.Consumption.Comb..mpg.,y=y,color="prediction"))

#regression using multiple inputs
dnn_model <- build_and_complie_model(normalizer)
summary(dnn_model)
history<-dnn_model %>% fit(as.matrix(train_features),as.matrix(train_labels),verbose=0,epochs=100,validation_split=0.2)
plot(history)
test_results[['dnn_model']]<- dnn_model %>% evaluate(as.matrix(test_features),as.matrix(test_labels),verbose=0)

#check performance for all models
sapply(test_results,function(x) x)

#make predictions
test_pred<-predict(dnn_model,as.matrix(test_features))
ggplot(data.frame(pred=as.numeric(test_pred),co2_emission=test_labels$CO2.Emissions.g.km.)) +
  geom_point(aes(x=pred,y=co2_emission)) +
  geom_abline(intercept = 0,slope = 1,color="blue")

#error distribution
qplot(test_pred-test_labels$CO2.Emissions.g.km.,geom="density")
error<-test_pred-test_labels

#save the model
save_model_tf(dnn_model,'dnn_model')
#save_model_hdf5(dnn_model,'dnn_model.h5')
#json_string<-model_to_json(dnn_model)
#write(json_string,"dnn_model_file.json")
#to reload the model
#reloaded<-load_model_tf('dnn_model')
#test_results[['reloaded']] <-reloaded %>% evaluate(as.matrix(test_features),as.matrix(test_labels),verbose=0)
#sapply(test_results,function(x) x)
co2_emission_predict<-function(x){
  data_new<-read.csv('CO2_Emissions.csv')
  x[length(x)+1]<-0
  data_new<-rbind(data_new,x)
  newdata<-data_new
  newdata$Engine.Size.L.<-as.numeric(newdata$Engine.Size.L.)
  newdata$Cylinders<-as.integer(newdata$Cylinders)
  newdata$Fuel.Consumption.City..L.100.km.<-as.numeric(newdata$Fuel.Consumption.City..L.100.km.)
  newdata$Fuel.Consumption.Hwy..L.100.km.<-as.numeric(newdata$Fuel.Consumption.Hwy..L.100.km.)
  newdata$Fuel.Consumption.Comb..L.100.km.<-as.numeric(newdata$Fuel.Consumption.Comb..L.100.km.)
  newdata$Fuel.Consumption.Comb..mpg.<-as.integer(newdata$Fuel.Consumption.Comb..mpg.)
  newdata$Fuel.Type=factor(newdata$Fuel.Type,labels = c(1,2,3,4,5))
  newdata$Fuel.Type=as.numeric(newdata$Fuel.Type)
  newdata$Make=as.numeric(factor(newdata$Make,labels=c(1:length(unique(newdata$Make)))))
  newdata$Model=as.numeric(factor(newdata$Model,labels=c(1:length(unique(newdata$Model)))))
  newdata$Vehicle.Class=as.numeric(factor(newdata$Vehicle.Class,labels=c(1:length(unique(newdata$Vehicle.Class)))))
  newdata$Transmission=as.numeric(factor(newdata$Transmission,labels=c(1:length(unique(newdata$Transmission)))))
  newdata<-recipe(CO2.Emissions.g.km. ~ .,newdata) %>% 
    step_num2factor(Fuel.Type,levels=c("Z","D","X","E","N")) %>%
    step_dummy(Fuel.Type,one_hot=TRUE) %>%
    prep() %>%
    bake(new_data= NULL)
  col_ind<-grep("CO2.Emissions.g.km.",colnames(newdata))
  pred<-newdata[nrow(newdata),-c(col_ind)]
  print(as.matrix(pred))
  View(newdata)
  val<-predict(dnn_model,as.matrix(pred))
  return(val)
}
#predict the single value
emitted_co2<-co2_emission_predict(c("ACURA","TL","MID-SIZE",3.5,6,"AS6","N",11.8,8.1,10.1,28))
cat("Emitted co2 is " , emitted_co2)


