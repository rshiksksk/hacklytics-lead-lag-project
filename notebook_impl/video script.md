2. 2-Minute Demo Video Script / Outline (Requirement 2)
Use this as a script and visual guide to record your demo video.

[0:00 - 0:25] The Problem & Hook: * Visual: A simple slide showing a supply chain (Company A -> Company B -> Company C).

Audio: "Information in the stock market isn't absorbed by everyone at once. If a raw material supplier faces a shortage, it takes time for the market to realize how this will impact their downstream customers. This is the 'Lead-Lag' effect. For Hacklytics, we built a machine learning pipeline to predict stock performance by mapping how information diffuses through the Chinese A-Share supply chain."

[0:25 - 1:00] How We Built It (The Tech): * Visual: Scroll through the lead-lag.ipynb Jupyter notebook, specifically highlighting the Data Preparation and Node2Vec training cells.

Audio: "We processed over 10 million rows of daily market data and built 10 annual supply chain graphs connecting both listed and private companies. To capture the graph structure without introducing look-ahead bias, we trained Node2Vec embeddings using PyTorch Geometric. We then fed these structural embeddings, along with a 20-day rolling window of market features, into a PyTorch LSTM model."

[1:00 - 1:40] The Results (Event Study): * Visual: Show the Classification Report matrix and then zoom in on the CAR (Cumulative Abnormal Return) line graph generated at the very end of your notebook.

Audio: "We evaluated our model on unseen data from 2024 to 2025. Standard accuracy metrics only tell part of the story, so we ran a financial Event Study. As you can see in our Cumulative Abnormal Returns plot, the model successfully identifies the lead-lag effect. Over 10 days, stocks we predicted to underperform (the red line) steadily drop by 0.7%, while predicted outperformers (the green line) trend positively. The information diffusion is real and tradable."

[1:40 - 2:00] Conclusion:

Visual: Team slide / GitHub repository link.

Audio: "By combining spatial graph embeddings with temporal LSTMs, our project proves that interconnected supply chain data holds massive predictive power for quantitative finance. Thank you!"