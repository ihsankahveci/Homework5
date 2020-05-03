#This file has two examples of supervised learning drawn from the Getting Started vignette
#https://journal.r-project.org/archive/2013-1/collingwood-jurka-boydstun-etal.pdf

#More detailed options (such as how to create the term document matrix from other data
#- unlike the example used here - can be found at:
https://cran.r-project.org/web/packages/RTextTools/RTextTools.pdf

#The first example below uses a single algorithm
#The second example uses an ensemble


library(RTextTools)
library(tm)


data(USCongress)
head(USCongress)


# CREATE THE DOCUMENT-TERM MATRIX
doc_matrix <- create_matrix(USCongress$text, language="english", removeNumbers=TRUE,stemWords=TRUE, removeSparseTerms=.998)

# SPLIT THE DATASET INTO TRAINING (first 4000) AND TESTING CASES (next 449)
container <- create_container(doc_matrix, USCongress$major, trainSize=1:4000,testSize=4001:4449, virgin=FALSE)

# SPECIFY THE ALGORITHM AND CLASSIFY

SVM <- train_model(container,"SVM")
SVM_CLASSIFY <- classify_model(container, SVM)

#COMPUTE ANALYTICS
analytics <- create_analytics(container,cbind(SVM_CLASSIFY))
topic_summary <- analytics@label_summary
alg_summary <- analytics@algorithm_summary
doc_summary <- analytics@document_summary 

#Confusion matrix (not part of RTextTools)

install.packages("caret")
library(caret)

confusion <- confusionMatrix(data = factor(analytics@document_summary$SVM_LABEL),
                reference = factor(analytics@document_summary$MANUAL_CODE))

#OUTPUT THE RESULTS

#make sure you know where the output is going
#getwd()
#setwd("C:/Users/John/Desktop")

write.csv(analytics@algorithm_summary, "AlgorithmSummary.csv") #precision, recall, f-score
write.csv(analytics@document_summary, "DocumentSummary.csv") #individual document predictions (could be used to contruct confusion matrix)
write.csv(confusion[["table"]], "confusion.csv")

##########################################################

#SECOND EXAMPLE compare performance of tWO algorithms and assesses agreement of the 'ensemble'
#Cross validation splits the sample into n different train and test sets. The average is reported as the algorithm performance.


library(RTextTools)
library(tm)

data(USCongress)
head(USCongress)

# CREATE THE DOCUMENT-TERM MATRIX
doc_matrix <- create_matrix(USCongress$text, language="english", removeNumbers=TRUE,stemWords=TRUE, removeSparseTerms=.998)

# SPLIT THE DATASET INTO TRAINING AND TESTING CASES
container <- create_container(doc_matrix, USCongress$major, trainSize=1:4000,testSize=4001:4449, virgin=FALSE)

SVM2 <- train_model(container,"SVM")
SVMcv <- cross_validate(container, 4, "SVM")
SVM2_CLASSIFY <- classify_model(container, SVM2)

#takes longer than SVM
RF2 <- train_model(container,"RF")
RFcv <- cross_validate(container, 4, "RF")
RF2_CLASSIFY <- classify_model(container, RF2)

analytics <- create_analytics(container,cbind(SVM2_CLASSIFY, RF2_CLASSIFY))

confusion <- confusionMatrix(data = factor(analytics@document_summary$SVM_LABEL),
                             reference = factor(analytics@document_summary$MANUAL_CODE))

#make sure you know the output is going



#Ensemble Summary: "Coverage" asks - how often do the n algorithms agree?
#"Recall asks" - how accurate is the prediction when n agree? 
write.csv(analytics@ensemble_summary, "EnsembleSummary.csv")

write.csv(analytics@label_summary, "TopicSummary.csv")
write.csv(analytics@algorithm_summary, "AlgorithmSummary.csv")
write.csv(analytics@document_summary, "DocumentSummary.csv")

                              
