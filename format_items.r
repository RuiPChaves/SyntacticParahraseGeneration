# Written by Rui P. Chaves

# Clean memory
rm(list=ls())

# Load dplyr
library(dplyr)


# ===========================================================================
# Ancillary functions
# ===========================================================================

# Basic clean up 
cleanup <- function(corpus) {
        
        # Select relevant columns
        corpus <- corpus %>% select(dyad,set,voice,construction,sentence)
        
        # Sort, enter ID values, and convert dyads to sequential integers
        corpus2 <- corpus %>% arrange(dyad,set,voice,construction,sentence) %>%
                mutate(dyad = as.numeric(dyad)) 
        
        # Remove original corpus from memory
        rm(corpus)
        
        # Rename dyad column
        corpus2  <- corpus2  %>% rename(item = dyad) 
        
 return(corpus2)
}


# Format cleaned dataset into BERT-ready format
bert_format <- function(corpus) {
        
        corpus$voice <- NULL
        corpus$construction <- NULL
        corpus <- corpus %>% select(item, set, sentence) 
        corpus <- droplevels(corpus)
        
        # Create raindom pairings, one per item
        bert_data <- corpus %>% 
                group_by(item) %>% 
                sample_n(size = 2, replace=TRUE) %>%
                group_by(item) %>% 
                mutate(Sentence = paste(sentence, collapse = ' [SEP] ')) %>%
                mutate(set = paste(set, collapse = "")) %>%
                ungroup()  %>%
                mutate(label = if_else(set == "aa" | set == "bb" | set == "cc", 1, 0)) %>%
                select(-set,-sentence) %>%
                distinct() %>%
                mutate(alpha = "a") %>%
                select(item, label, alpha, Sentence)
        
        return(bert_data)
        
}




# =====================  Clean test dataset  ==========================================
corpus_test <- cleanup(read.csv("~/DeepL/Paraphrase_2/sentence_generator.test.csv"))

# ===================  Convert test dataset into BERT-ready format  ==================
bert_data_test <- bert_format(corpus_test)

# ===================  Write out test dataset in BERT format  ==================
write.table(bert_data_test, file = "~/DeepL/Paraphrase_2/test_r3.tsv", 
            sep = "\t", col.names = FALSE, row.names = FALSE, quote = FALSE)

rm(corpus_test)
rm(bert_data_test)


# =====================  Clean train dataset  ==========================================
corpus_train <- cleanup(read.csv("~/DeepL/Paraphrase_2/sentence_generator.train.csv"))

# ===================  Convert test dataset into BERT-ready format  ==================
bert_data_train <- bert_format(corpus_train)

# ===================  Write out test dataset in BERT format  ==================
write.table(bert_data_train, file = "~/DeepL/Paraphrase_2/train_r3.tsv", 
            sep = "\t", col.names = FALSE, row.names = FALSE, quote = FALSE)

rm(corpus_train)
rm(bert_data_train)
