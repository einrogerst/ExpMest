# Este script usa o "pang lee sentiment dataset" e transforma eles em n conjuntos para treinamento e teste 
# de classificação de polarização de sentimentos utilizando a biblioteca foreach para paralelização.
# As Matrizes Termo-Documento são criadas usando a biblioteca tm, que tira proveito de matrizes esparsas 
# para sua armazenagem. Os valores de TF-IDF nos conjuntos de teste são obtidos através dos valores
# de IDF do conjuto de treinamento.
# 

require(foreach)
require(doParallel)
logical.cores <- detectCores(logical=T)
cl <- makeCluster(logical.cores)
registerDoParallel(cl) 
library(tm)
library(slam)

setwd("C:/Users/i826950/Desktop/Personal/panglee")
options(stringsAsFactors = FALSE)
set.seed(7)

getTrainTestCorpus <- function(dirSrcFolds, dirSrc, i, basicControl, myStopwordsList) {
  #initialize a DirSource for Training
  dirSrcTrain <- dirSrcFolds[[i]]
  #Fill the filelist with all files that don't belong to the current fold
  dirSrcTrain$filelist <- dirSrc$filelist[-folds[[i]]]
  dirSrcTrain$length <- length(dirSrcTrain$filelist)
  #build a Corpus with the training set from the DirSource
  trainCorpus <- VCorpus(dirSrcTrain)
  #construct a Document Term matrix (DTM) from the training Corpus
  trainDtm <-  
    DocumentTermMatrix(trainCorpus,
                       control=c(basicControl,
                                 weighting = function(x) weightTfIdf(x)))
  
  #save the IDF for latter use during test
  trainIDF <- log2(nDocs(trainDtm)/col_sums(trainDtm > 0))
  trainIDF[!is.finite(trainIDF)] <- 0
  
  #add the class to the DTM
  trainDtm$docsentclass <- substr(dirSrcTrain$filelist, 16, 18)

  ###### attribute selection goes here ####################################################################
  # library(FSelector)
  # #calculate the Information Gain of each attribute (term) from the training DTM
  # IGtemp <- as.data.frame(as.matrix(trainDtm))
  # IGtemp$docsentclass <- trainDtm$docsentclass
  # infogains <-
  #   for(i=1:length(IGtemp)
  #     result <- FSelector::information.gain(docsentclass~., IGtemp[,c(i, length(IGtemp))])
  #   
  # 
  # i=200
  # #select only the termsToKeep best attributes
  # #bestAttrs <- cutoff.k(infogains, termsToKeep)
  # 
  # print(paste0("Number of terms with no Information Gain in fold ", i, ": ", sum(infogains[bestAttrs,]==0), "/", termsToKeep))
  # #restrict the DTM to the best attributes only
  # #trainDtm <- trainDtm[,c(bestAttrs, "docsentclass")]
  # ### ._ don't forget to adjust the trainIDF too!!
  
  #create a test DirSource
  dirSrcTest <- dirSrcFolds[[i]]
  #build a test Corpus from test DirSource
  testCorpus <- VCorpus(dirSrcFolds[[i]])
  
  #create a test corpus using the IDFs previously calculated
  testDtm <- 
    DocumentTermMatrix(testCorpus, 
                       control=c(basicControl,
                                 weighting = function(x) weightTfForeignIdf(x, normalize = TRUE, trainIDF)))

  testDtm$docsentclass <- substr(dirSrcTest$filelist, 16, 18)

  list(list(round=i, train=trainDtm, test=testDtm))
  
}

#tf-idf using IDF from training
weightTfForeignIdf <- 
  WeightFunction(
    function (m, normalize = TRUE, IDF) 
    {
      require(slam)
      isDTM <- inherits(m, "DocumentTermMatrix")
      if (isDTM) 
        m <- t(m)
      if (normalize) {
        cs <- col_sums(m)
        if (any(cs == 0)) 
          warning("empty document(s): ", paste(Docs(m)[cs == 
                                                         0], collapse = " "))
        names(cs) <- seq_len(nDocs(m))
        m$v <- m$v/cs[m$j]
      }
      #consider terms that exist in both the DTM and the IDF only
      m <- m[intersect(rownames(m), names(IDF)),]
      
      #add other terms from IDF to the matrix
      zeroes <- simple_triplet_zero_matrix(nrow=length(setdiff(names(IDF), rownames(m))), ncol=ncol(m))
      rownames(zeroes) <- setdiff(names(IDF), rownames(m))
      m <- rbind(m, zeroes)
      m <- m[order(rownames(m)),]
      class(m) <- c("TermDocumentMatrix", "simple_triplet_matrix")
      
      #multiply the TF by the IDF
      m <- m * IDF
      
      attr(m, "weighting") <- c(sprintf("%s%s", "TF-IDF using internal", 
                                        if (normalize) " (normalized)" else ""), "tf-idf and foreign IDF")
      if (isDTM) 
        t(m)
      else m
    },
    "tfForeignIdf", "tfForeignIdf")

if (!file.exists("./txt_sentoken/.")){
  download.file("http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz", destfile="review_polarity.tar.gz")  
  untar("review_polarity.tar.gz")
}

dirSrc <- DirSource(directory = "./txt_sentoken",
          encoding = "UTF-8",
          recursive = TRUE)

#limit the sources for test purposes (remove it in production)
#dirSrc$length <- 100
#dirSrc$filelist <- sample(dirSrc$filelist, 100)

library(caret)
numFolds <- 10
folds <- createFolds(1:length(dirSrc$filelist), k=numFolds)
dirSrcFolds <- 
  lapply(folds,
         function(x){
           temp <- dirSrc
           temp$length <- length(x)
           temp$filelist <- temp$filelist[x]
           temp}
         )

myStopwordsList <-
  read.table("./stopwords.txt", 
             sep="\n", 
             fill=FALSE, 
             strip.white=TRUE, 
             encoding = "UTF-8", 
             quote="",
             stringsAsFactors = FALSE)

basicControl <- list(removePunctuation = TRUE,
                  removeNumbers = TRUE,
                  stemming = TRUE,
                  stopwords = myStopwordsList,
                  wordLengths = c(3, Inf))

system.time(corpusFolds <- 
  foreach(i = 1:numFolds, .combine=c, .packages=c("tm", "slam")) %dopar% 
  getTrainTestCorpus(dirSrcFolds, dirSrc, i, basicControl, myStopwordsList))

################ END OF PREPROCESSING ########################

i=1
library(Matrix)
library(xgboost)
train <- 
  xgb.DMatrix(
    sparseMatrix(i=corpusFolds[[i]]$train$i, 
                 j=corpusFolds[[i]]$train$j, 
                 x=corpusFolds[[i]]$train$v, 
                 dims=c(corpusFolds[[i]]$train$nrow, corpusFolds[[i]]$train$ncol), 
                 dimnames = corpusFolds[[i]]$train$dimnames),
    label=as.numeric(corpusFolds[[i]]$train$docsentclass=="pos"))

xg1 <- xgb.train(params=list(booster="gbtree", 
                             max_depth=6,
                             objective="binary:logistic",
                             subsample=1,
                             eta=0.01),
                 data=train, 
                 nrounds=500,
                 verbose = 2,
                 nthread = 2,
                 eval_metric="error")

xg_simple <- xgboost(data=train, nrounds=1000, eta=0.01, max_depth=3)
print(xg_simple)
# xgrf <- xgb.train(params=list(booster="gbtree", 
#                              max_depth=Inf,
#                              objective="binary:logistic",
#                              eta=0.5,
#                              subsample=0.632,
#                              colsample_bytree=floor(sqrt(corpusFolds[[i]]$train$ncol)), 
#                              num_parallel_tree=200,
#                              base_score=0.5),
#                  data=train, 
#                  nrounds=1,
#                  verbose = 2,
#                  nthread = 2,
#                  eval_metric="error")

test <- 
  xgb.DMatrix(
    sparseMatrix(i=corpusFolds[[i]]$test$i, 
                 j=corpusFolds[[i]]$test$j, 
                 x=corpusFolds[[i]]$test$v, 
                 dims=c(corpusFolds[[i]]$test$nrow, corpusFolds[[i]]$test$ncol), 
                 dimnames = corpusFolds[[i]]$test$dimnames),
    label=as.numeric(corpusFolds[[i]]$test$docsentclass=="pos"))

sum(as.numeric(corpusFolds[[i]]$train$docsentclass=="pos")==as.numeric(predict(xg_simple, train)>0.5))/nrow(corpusFolds[[i]]$train)

sum(as.numeric(corpusFolds[[i]]$test$docsentclass=="pos")==as.numeric(predict(xg_simple, test)>0.5))/nrow(corpusFolds[[i]]$test)
  
hist(predict(xg1, test))
hist(predict(xg1, train))

accXGboost <- function(trainData, testData, testLabels, pNrounds, pEta, pMax_depth, pSubsample)
{
  print(paste0("Rounds: ", floor(pNrounds), ", Eta: ", pEta, ", Subsample: ", pSubsample, ", Max Depth: ", floor(pMax_depth)))
  myModel <- 
    xgb.train(params = list(booster = "gbtree",
                            eta=pEta, 
                            max_depth = floor(pMax_depth),
                            subsample = pSubsample,
                            objective = "binary:logistic"),
              data=trainData, 
              nrounds=floor(pNrounds), 
              verbose = 0)
  sum(testLabels==as.numeric(predict(myModel, testData)>0.5))/length(testLabels)
}

# accXGboost(trainData = train,
#            testData = test,
#            testLabels=as.numeric(corpusFolds[[i]]$test$docsentclass=="pos"),
#            pNrounds = 5,
#            pEta = 1,
#            pMax_depth = 3)

library(GA)
GA <- ga(type = "real-valued", 
         fitness = function(x) accXGboost(trainData = train,
                                          testData = test,
                                          testLabels=as.numeric(corpusFolds[[i]]$test$docsentclass=="pos"),
                                          pNrounds=x[1],
                                          pEta=x[2],
                                          pMax_depth=x[3], 
                                          pSubsample=x[4]),
         names = c("pNrounds", "pEta", "pMax_depth", "pSubsample"),
         min = c(1, 0.0001, 1, 0.1), 
         max = c(100, 1, 30, 1),
         popSize = 50, 
         maxiter = 30)

summary(GA)
GA@summary
summary(lm(X4~., data=data.frame(cbind(GA@population, GA@fitness))))
