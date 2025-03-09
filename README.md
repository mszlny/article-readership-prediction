Article Readership Prediction Model

This project predicts the number of clicks (readership) an article title will generate, based on real-world data from a news portal.

Dataset Description
The training data was scraped from a real news portal’s admin panel, so it is not publicly available. However, I have attached a sample Excel file that shows the dataset structure.

The columns in the dataset:

•	Date – Publication date of the article.

•	Title – Article title.

•	URL – Link to the article.

•	AI-generated? – Whether the article was written by AI or not.

•	Stat – Number of clicks (readership count).

•	Heading – Section/category of the article.

•	Author – Name of the article’s author.

•	Top – A number from 0 to 4 indicating its position on the homepage:
      o	0: Not featured (appears only among the latest articles).
      o	1: Most prominently featured at the top of the homepage.
      o	2–4: Other featured articles at the top of the homepage. (However, this ranking has limited significance, as most readership comes from news aggregators, Google Discovery and Facebook.)
      
•	Lead article? – Whether the article is a lead article or not. Larger, original pieces are marked as lead articles based on the time of day.

The model is specifically trained on this news portal’s data. To make a prediction, the title and the lead article type must be provided.

IMPORTANT!
This project is an experimental hobby project. The model’s predictions are currently imprecise and require further refinement. However, predicting readership is inherently challenging, as it is largely influenced by external factors such as Google and Facebook algorithms, which determine article visibility.

It is also important that the 'preprocess_article' function does not work because I did not upload the trained vectorizer.

Feel free to explore the code! Feedback is always welcome. 😊
