# require(foreach)
# require(doParallel)
# logical.cores <- detectCores(logical=T)
# cl <- makeCluster(logical.cores)
# registerDoParallel(cl) 

setwd("C:/Users/i826950/Desktop/Personal/panglee")
options(stringsAsFactors = FALSE)
set.seed(7)

library(tm)
dirSrc <- DirSource(directory = "./txt_sentoken",
          encoding = "UTF-8",
          recursive = TRUE)

#limit the sources for test purposes (remove it in production)
dirSrc$length <- 100
dirSrc$filelist <- sample(dirSrc$filelist, 100)

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

termsToKeep = 10
library(FSelector)
i=2 

for(i in 1:numFolds){
  #initialize a DirSource for Training
  dirSrcTrain <- dirSrcFolds[[i]]
  #Fill the filelist with all files that don't belong to the current fold
  dirSrcTrain$filelist <- dirSrc$filelist[-folds[[i]]]
  dirSrcTrain$length <- length(dirSrcTrain$filelist)
  #build a Corpus with the training set from the DirSource
  trainCorpus <- VCorpus(dirSrcTrain)
  #construct a Document Term matrix (DTM) from the training Corpus
  trainDtm <- as.data.frame(as.matrix(
    DocumentTermMatrix(trainCorpus, 
                       control=c(basicControl,
                                 weighting = function(x) weightTf(x)))))
  #create an IDF for later use with the test set
  trainIDF <- log2(nrow(trainDtm)/colSums(trainDtm>0))
  names(trainIDF) <- names(trainDtm)
  #add the class to the DTM
  trainDtm$docsentclass <- substr(dirSrcTrain$filelist, 16, 18)
  #calculate the Information Gain of each attribute (term) from the training DTM
  infogains <- information.gain(docsentclass~., trainDtm)
  #select only the termsToKeep best attributes 
  bestAttrs <- cutoff.k(infogains, termsToKeep)
  print(paste0("Number of terms with no Information Gain in fold ", i, ": ", sum(infogains[bestAttrs,]==0), "/", termsToKeep))
  #restrict the DTM to the best attributes only
  trainDtm <- trainDtm[,c(bestAttrs, "docsentclass")]

  #create a test DirSource
  dirSrcTest <- dirSrcFolds[[i]]
  #build a test Corpus from test DirSource
  testCorpus <- VCorpus(dirSrcFolds[[i]])
  
  testDtm <- as.data.frame(as.matrix(
    DocumentTermMatrix(testCorpus, 
                       control=c(basicControl,
                                 weighting = function(x) weightTfNorm(x)))))
  
  head(testDtm[, names(trainIDF)])
  dim(merge(testDtm, trainIDF))
  
  sweep
  apply(testDtm, 1, function(x){
    x
  })

  
  med.att <- apply(attitude, 2, median)
  sweep(data.matrix(attitude), 2, med.att)  # subtract the column medians
  
  
  (testDtm*trainIDF)[, "wow"]
  
  head(as.data.frame(trainIDF))
  
  library(reshape2)
  meltTestDtm <- add_rownames(testDtm, var = "document")
  meltTestDtm <- melt(meltTestDtm, variable.name="term", value.name="tf")
  merge
  table
  trainIDF
  testDtm$docsentclass <- substr(dirSrcFolds[[i]]$filelist, 16, 18)
    
  testDtm <- testDtm[,names(trainDtm)]
  
  infogains[,]
  infogains
  testCorpus <- VCorpus(dirSrcFolds[[i]], readerControl = list(language = "en"))
  
  experiment <- list(trainCorpus=trainCorpus, trainDtm=trainDtm)
  experimentRounds <- c(experimentRounds, experiment)
  dtmFolds[[i]] <- dtm
}

DocumentTermMatrix
dtmDF <- as.data.frame(as.matrix(dtm))
dtmDF$docsentclass[1:1000] <- 'neg'
dtmDF$docsentclass[1001:2000] <- 'pos'
dtmDF$docsentclass <- as.factor(dtmDF$docsentclass)

#library(foreign)
#write.csv(crudeDTM, "movie review docterm matrix.csv")

library(FSelector)
if(file.exists("infogains.RData")) {
  load("infogains.RData")
} else {
  system.time(infogains1 <- information.gain(docsentclass~., dtmDF[c(1:10000, length(dtmDF))]))
  system.time(infogains2 <- information.gain(docsentclass~., dtmDF[c(10001:20000, length(dtmDF))]))
  system.time(infogains3 <- information.gain(docsentclass~., dtmDF[20001:length(dtmDF)]))
  infogains <- rbind(infogains1, infogains2, infogains3)
  save(infogains, file="infogains.RData")
}


subset <- subset(infogains, attr_importance > 0)
#118?!?!

dtm.good.ig <- dtmDF[,c(rownames(subset), "docsentclass")]

##### Feed-Forward Neural Networks
library(nnet)

num.folds <- 10
folds <- createFolds(dtm.good.ig[,1], k=num.folds)
acc <- numeric()
for(i in 1:num.folds){
  train.idx <- setdiff(1:2000, unlist(folds[i]))
  test.idx <- unlist(folds[i])
  nn1 <- nnet(docsentclass~., data=dtm.good.ig[train.idx,], 
              size = 3, rang = 0.5, decay = 5e-3, maxit = 1000, trace=F)

  conf.mtx <- table(
    data.frame(
      pred=predict(nn1, dtm.good.ig[test.idx, 1:length(dtm.good.ig)-1], type="class"),
      act=dtm.good.ig[test.idx, "docsentclass"]))
    
  acc[i] <- (conf.mtx[1]+conf.mtx[4]) / sum(conf.mtx)
}
mean(acc)

##### Back-propagation Neural Networks
library(neuralnet)

dtm.good.ig$docsentclass <- factor(dtm.good.ig$docsentclass, labels=c(0,1))

num.folds <- 10
folds <- createFolds(dtm.good.ig[,1], k=num.folds)

acc <- numeric()
for(i in 1:num.folds){
  train.idx <- setdiff(1:2000, unlist(folds[i]))
  test.idx <- unlist(folds[i])

  dtm.good.ig
  nn1 <- 
    neuralnet(
      formula=as.formula(paste("docsentclass", paste(names(dtm.good.ig[1:length(dtm.good.ig)-1]), collapse="+"), sep="~")), 
      data=as.matrix(dtm.good.ig[train.idx,]))
  conf.mtx <- table(
    data.frame(
      pred=predict(nn1, dtm.good.ig[test.idx, 1:length(dtm.good.ig)-1], type="class"),
      act=dtm.good.ig[test.idx, "docsentclass"]))
  
  acc[i] <- (conf.mtx[1]+conf.mtx[4]) / sum(conf.mtx)
}


##### tests with randomForest
library(randomForest)

  rf <- randomForest(docsentclass~., data=dtm.good.ig, importance=F, proximity=F, ntree=100)


compare0 <- 
  as.data.frame(
    cbind(
      predic=predict(rf, dtm.good.ig[,1:(length(dtm.good.ig)-1)], type="response"),
      predict(rf, dtm.good.ig[,1:(length(dtm.good.ig)-1)], type="prob"), 
      actual=as.character(dtm.good.ig$docsentclass)))

table(compare0$predic, compare0$actual)
compare0$predic== & compare0$pos
summary(compare0)
pred <- prediction( as.numeric(compare0$pos), factor(compare0$predic, labels=c(0,1)))
perf <- performance( pred, "tpr", "fpr" )
plot(perf)


#rf100 <- foreach(numtree=25, .combine=combine, .packages='randomForest') %dopar% randomForest(x=crudeDTM[,1:(length(crudeDTM)-1)] , y=crudeDTM$docsentclass, importance=F, proximity=F, ntree=numtree)

library(maxent)

sparse <- as.compressed.matrix(dtm.good.ig[,1:(length(dtm.good.ig)-1)])
num.folds <- 10
folds <- createFolds(dtm.good.ig[,1], k=num.folds)
acc <- numeric()
for(i in 1:num.folds){
  train.idx <- setdiff(1:2000, unlist(folds[i]))
  test.idx <- unlist(folds[i])

  maxent1 <- maxent(sparse[train.idx,], dtm.good.ig[train.idx, "docsentclass"])

  conf.mtx <- table(
    data.frame(
      pred=factor(predict(maxent1, dtm.good.ig[test.idx, 1:length(dtm.good.ig)-1])[,1]),
      act=dtm.good.ig[test.idx, "docsentclass"]))
  
  acc[i] <- (conf.mtx[1]+conf.mtx[4]) / sum(conf.mtx)
}
mean(acc)


compare1 <- as.data.frame(cbind(predict(maxent1, dtm.good.ig[,1:(length(dtm.good.ig)-1)]), as.character(dtm.good.ig$docsentclass)))
table(compare1$labels,compare1$V4)


library(ROCR)
pred <- prediction( as.numeric(compare1$pos), as.numeric(factor(compare1$V4, labels=c(0,1))))
perf <- performance( pred, "tpr", "fpr" )
plot(perf)

library(cvTools)

library(ineq)
Gini(dtm.good.ig[,1])

sapply(dtm.good.ig, Gini)

Gini
me.call <- call("maxent", formula = Y ~ .)
# perform cross-validation
cvFit(call, data = coleman, y = coleman$Y, cost = rtmspe, 
      K = 5, R = 10, costArgs = list(trim = 0.1), seed = 1234)

weightTfNorm <- 
  WeightFunction(
    function (m) {
      require(slam)
      isDTM <- inherits(m, "DocumentTermMatrix")
      if (isDTM) 
        m <- t(m)
      cs <- col_sums(m)
      if (any(cs == 0)) 
        warning("empty document(s): ", paste(Docs(m)[cs == 
                                                       0], collapse = " "))
      names(cs) <- seq_len(nDocs(m))
      m$v <- m$v/cs[m$j]
      if (isDTM) 
        t(m)
      else m
    },
    "tfNorm", "tfNorm")

#tf-idf using IDF from training
weightTfForeignIdf <- 
  WeightFunction(
    function (m, normalize = TRUE, IDF) 
    {
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
      asdfasdf
      rs <- row_sums(m > 0)
      if (any(rs == 0)) 
        warning("unreferenced term(s): ", paste(Terms(m)[rs == 
                                                           0], collapse = " "))
      lnrs <- log2(nDocs(m)/rs)
      lnrs[!is.finite(lnrs)] <- 0
      m <- m * lnrs
      attr(m, "weighting") <- c(sprintf("%s%s", "term frequency - inverse document frequency", 
                                        if (normalize) " (normalized)" else ""), "tf-idf")
      if (isDTM) 
        t(m)
      else m
    },
    "tfForeignIdf", "tfForeignIdf")

#my own log normalized tf-idf
weightLogTfIdf <- 
  WeightFunction(
    function(m){
      require(slam)
      isDTM <- inherits(m, "DocumentTermMatrix")
      if (isDTM) 
        m <- t(m)
      m$v <- log1p(m$v)
      rs <- row_sums(m > 0)
      if (any(rs == 0)) 
        warning("unreferenced term(s): ", paste(Terms(m)[rs == 
                                                           0], collapse = " "))
      lnrs <- log2(nDocs(m)/rs)
      lnrs[!is.finite(lnrs)] <- 0
      m <- m * lnrs
      attr(m, "weighting") <- "log normalized tf-idf"
      if (isDTM) 
        t(m)
      else m
    },
    "log normalized tf-idf", "lognormtfidf")