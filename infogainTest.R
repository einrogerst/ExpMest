setwd("C:/Users/i826950/Desktop/Personal/panglee")
library(tm)
library(FSelector)
require(foreach)
require(doParallel)

set.seed(7)
options(stringsAsFactors = FALSE)
logical.cores <- detectCores(logical=T)
cl <- makeCluster(logical.cores)
registerDoParallel(cl) 

if (!file.exists("./txt_sentoken/.")){
  download.file("http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz", destfile="review_polarity.tar.gz")  
  untar("review_polarity.tar.gz")
}

dirSrc <- DirSource(directory = "./txt_sentoken",
                    encoding = "UTF-8",
                    recursive = TRUE)

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

myCorpus <- VCorpus(dirSrc)
#construct a Document Term matrix (DTM) from the training Corpus
myDtm <- as.data.frame(as.matrix(
  DocumentTermMatrix(myCorpus, 
                     control=c(basicControl,
                               weighting = function(x) weightTfIdf(x)))))
#add the class to the DTM
myDtm$docsentclass <- substr(dirSrc$filelist, 16, 18)

# myDtm2 <- DocumentTermMatrix(myCorpus, 
#                      control=c(basicControl,
#                                weighting = function(x) weightTfIdf(x)))
# #add the class to the DTM
# myDtm2$docsentclass <- substr(dirSrc$filelist, 16, 18)

#calculate the Information Gain of each attribute (term) from the training DTM
infoGains <-
  foreach(i=1:(ncol(myDtm)-1), .combine=rbind, .packages='FSelector') %dopar%
    information.gain(docsentclass~., myDtm[c(i, length(myDtm))])

# system.time(infoGainsMtx <- foreach(i=1:ncol(myDtm2), 
#                         .combine=rbind, 
#                         .packages=c('FSelector', 'tm')) 
#                 %dopar%
#               information.gain(docsentclass~., 
#                                as.data.frame(
#                                  cbind(as.matrix(myDtm2[, i]), 
#                                        docsentclass=myDtm2$docsentclass))))

save(infoGains, file="infogainsTestTfIdf.RData")
nonZeroIGs <- infoGains[infoGains$attr_importance>0,,drop=F]
nonZeroIGs <- nonZeroIGs[order(nonZeroIGs$attr_importance, decreasing = T), , drop = F]
write.csv2(nonZeroIGs, file = "infoGainsTfIdfNorm.csv")

##### TF
myDtm <- as.data.frame(as.matrix(
  DocumentTermMatrix(myCorpus, 
                     control=c(basicControl,
                               weighting = function(x) weightTf(x)))))

#add the class to the DTM
myDtm$docsentclass <- substr(dirSrc$filelist, 16, 18)
#calculate the Information Gain of each attribute (term) from the training DTM

infoGains <-
  foreach(i=1:(ncol(myDtm)-1), .combine=rbind, .packages='FSelector') %dopar%
  information.gain(docsentclass~., myDtm[c(i, length(myDtm))])

save(infoGains, file="infogainsTestTf.RData")
nonZeroIGs <- infoGains[infoGains$attr_importance>0,,drop=F]
nonZeroIGs <- nonZeroIGs[order(nonZeroIGs$attr_importance, decreasing = T), , drop = F]
write.csv2(nonZeroIGs, file = "infoGainsTf.csv")

##### Binary 
myDtm <- as.data.frame(as.matrix(
  DocumentTermMatrix(myCorpus, 
                     control=c(basicControl,
                               weighting = function(x) weightBin(x)))))

#add the class to the DTM
myDtm$docsentclass <- substr(dirSrc$filelist, 16, 18)
#calculate the Information Gain of each attribute (term) from the training DTM

system.time(infoGains <-
  foreach(i=1:(ncol(myDtm)-1), .combine=rbind, .packages='FSelector') %dopar%
  information.gain(docsentclass~., myDtm[c(i, length(myDtm))]))

save(infoGains, file="infogainsTestBin.RData")
nonZeroIGs <- infoGains[infoGains$attr_importance>0,,drop=F]
nonZeroIGs <- nonZeroIGs[order(nonZeroIGs$attr_importance, decreasing = T), , drop = F]
write.csv2(nonZeroIGs, file = "infoGainsBin.csv")

#select only the termsToKeep best attributes 
#bestAttrs <- cutoff.k(infogains, termsToKeep)

