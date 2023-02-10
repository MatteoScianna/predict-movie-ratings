# predict-movie-ratings
## Inspired from the article "Film Success Prediction Using NLP Techniques" by Joshua A Gross, William C Roberson, J Bay Foley-Cox, this project aims to predict movie ratings and movie revenues using NLP applied to movie scripts.

Starting from a pool of circa 1000 movie scripts scraped from https://imsdb.com/, dict_features is populated by assigning to each movie a set of predictors. From the movie script itself, top250 words, together with some indices of lexical and semantical complexity are extracted. Furthermore, thanks to IMDb API some movie information such as run-time, genres, year of release and budget were assign. File "collection_and_preprocessing.py" deals with the creation of this dictionary (available in the folder "dict"), together with two other dictionaries that will be used by the model. 
Note that some manual labelling has been done in order to complete the dataset.
<br>
File "ml_model.py" is an adaptation of the model of the article cited above, in order to take into account also NLP features extracted in this context. 
The model predicts log(revenue) and rating of a movie. 

## References

<br> [1] Gross, Joshua A., William C. Roberson, and J. Bay Foley-Cox. "Cs 230: film success prediction using nlp techniques." (2021).
<br> [2] E. Castano et al. “On the Complexity of Literary and Popular Fiction (Under revision)

