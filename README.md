# predict-movie-ratings
## Inspired from the article "Film Success Prediction Using NLP Techniques" by Joshua A Gross, William C Roberson, J Bay Foley-Cox, this project aims to predict movie ratings and movie revenues using NLP applied to movie scripts.

Starting from a pool of circa 1000 movie scripts scraped from https://imsdb.com/ , dict_features is populated by assigning to each movie a set of predictors. From the movie script itself, top250 words, together with some indices of lexical and semantical complexity are extracted. Furthermore, thanks to IMDb API some movie information such as run-time, genres, year of release and budget were assign. 

By adapting the model of the article cited above, movie revenues and IMDb movie ratings are predicted. 
