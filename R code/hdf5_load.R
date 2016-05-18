# http://www.bioconductor.org/packages/release/bioc/vignettes/rhdf5/inst/doc/rhdf5.pdf
# install.packages("ggplot2")


h5f = H5Fopen("..\\..\\expedia_data\\data_test.h5")


data<-loadhdf5data("..\\..\\expedia_data\\data_test.h5")

library(rhdf5)
library(bit64)



loadhdf5data <- function(h5File) {
     
     listing <- h5ls(h5File)
     # Find all data nodes, values are stored in *_values and corresponding column
     # titles in *_items
     data_nodes <- grep("_values", listing$name)
     name_nodes <- grep("_items", listing$name)
   
     data_paths = paste(listing$group[data_nodes], listing$name[data_nodes], sep = "/")
     name_paths = paste(listing$group[name_nodes], listing$name[name_nodes], sep = "/")
     
     columns = list()
     for (idx in seq(data_paths)) {
         data <- data.frame(t(h5read(h5File, data_paths[idx] ,bit64conversion="bit64" )))
         names <- t(h5read(h5File, name_paths[idx] ,bit64conversion="bit64" ))
         entry <- data.frame(data)
         colnames(entry) <- names
         columns <- append(columns, entry)
     }
     
     data <- data.frame(columns)
   
     return(data)
}