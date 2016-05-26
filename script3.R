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

cl <- makeCluster(logical.cores)
registerDoParallel(cl) 

system.time(corpusFolds <- 
  foreach(i = 1:numFolds, .combine=c, .packages=c("tm", "slam")) %dopar% 
  getTrainTestCorpus(dirSrcFolds, dirSrc, i, basicControl, myStopwordsList))

################ END OF PREPROCESSING ########################

library(Matrix)
library(xgboost)
# i=1
# train <- 
#   xgb.DMatrix(
#     sparseMatrix(i=corpusFolds[[i]]$train$i, 
#                  j=corpusFolds[[i]]$train$j, 
#                  x=corpusFolds[[i]]$train$v, 
#                  dims=c(corpusFolds[[i]]$train$nrow, corpusFolds[[i]]$train$ncol), 
#                  dimnames = corpusFolds[[i]]$train$dimnames),
#     label=as.numeric(corpusFolds[[i]]$train$docsentclass=="pos"))

system.time(
train <- lapply(corpusFolds, function(corpusFold) 
   tr <- xgb.DMatrix(
    sparseMatrix(i=corpusFold$train$i,
                 j=corpusFold$train$j,
                 x=corpusFold$train$v,
                 dims=c(corpusFold$train$nrow, corpusFold$train$ncol),
                 dimnames = corpusFold$train$dimnames),
    label=as.numeric(corpusFold$train$docsentclass=="pos")))
)

system.time(
  test <- lapply(corpusFolds, function(corpusFold) 
    tr <- xgb.DMatrix(
      sparseMatrix(i=corpusFold$test$i,
                   j=corpusFold$test$j,
                   x=corpusFold$test$v,
                   dims=c(corpusFold$test$nrow, corpusFold$test$ncol),
                   dimnames = corpusFold$test$dimnames),
      label=as.numeric(corpusFold$test$docsentclass=="pos")))
)

#
# xg_simple <- xgboost(data=train, nrounds=1000, eta=0.01, max_depth=3)
#
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


sum(as.numeric(corpusFolds[[i]]$train$docsentclass=="pos")==as.numeric(predict(xg_simple, train)>0.5))/nrow(corpusFolds[[i]]$train)

sum(as.numeric(corpusFolds[[i]]$test$docsentclass=="pos")==as.numeric(predict(xg_simple, test)>0.5))/nrow(corpusFolds[[i]]$test)
  
hist(predict(xg1, test))
hist(predict(xg1, train))

logDF <- data.frame()
accXGboost <- function(trainData, testData, testLabels, pNrounds, pEta, pMax_depth, pSubsample, pFold)
{
  myModel <- 
    xgb.train(params = list(booster = "gbtree",
                            eta=pEta, 
                            max_depth = floor(pMax_depth),
                            subsample = pSubsample,
                            objective = "binary:logistic"),
              data=trainData, 
              nrounds=floor(pNrounds), 
              verbose = 0)
  result <- sum(testLabels==as.numeric(predict(myModel, testData)>0.5))/length(testLabels)
  print(paste0("Fold: ", pFold,
               ", Rounds: ", floor(pNrounds),
               ", Eta: ", pEta,
               ", Subsample: ", pSubsample,
               ", Max Depth: ", floor(pMax_depth),
               ", Accuracy: ", result))
  logDF <<- rbind(logDF, 
                  data.frame(Fold=pFold, 
                             Rounds = floor(pNrounds), 
                             Eta = pEta,
                             Subsample = pSubsample,
                             MaxDepth = floor(pMax_depth),
                             Accuracy = result))
  result
}

# accXGboost(trainData = train,
#            testData = test,
#            testLabels=as.numeric(corpusFolds[[i]]$test$docsentclass=="pos"),
#            pNrounds = 5,
#            pEta = 1,
#            pMax_depth = 3)

library(GA)

GA <- ga(type = "real-valued", 
         fitness = function(x) accXGboost(trainData = train[[floor(x[1])]],
                                          testData = test[[floor(x[1])]],
                                          testLabels=as.numeric(corpusFolds[[floor(x[1])]]$test$docsentclass=="pos"),
                                          pNrounds=x[2],
                                          pEta=x[3],
                                          pMax_depth=x[4], 
                                          pSubsample=x[5],
                                          pFold=floor(x[1])),
         names = c("Fold", "Rounds", "Eta", "Max_depth", "Subsample"),
         min = c(    1,   1, 0.0001,  1, 0.1), 
         max = c(10.99, 100,      1, 30,   1),
         popSize = 60, 
         maxiter = 30)

save(logDF, file = "logGA.RData")
save(GA, file = "GA.RData")

summary(GA)
GA@summary
summary(lm(X6~., data=data.frame(cbind(GA@population, GA@fitness))))

summary(logDF[logDF$Accuracy>0.85,-1])

library(reshape2)
str(logDF)
graphData <- melt(logDF, id.vars = "Accuracy")
graphData <- graphData[graphData$variable!="Fold",]
ggplot(data=graphData, mapping=aes(x=Accuracy, y=value)) + 
  geom_bin2d(bins=25) +
  #geom_jitter(shape=1) + 
  facet_wrap( ~ variable, ncol=2, scales="free_y") + scale_colour_gradient(trans = "log")


