dip_test <- function(vector, full.result = FALSE, min.is.0 = FALSE, debug = FALSE) {
  library(diptest)
  return(dip(vector))
}
