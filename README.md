# Book Recommendation System

This code provides a system to recommend books based on user input. The dataset used is books.csv.

## Python Libraries used

- numpy
- pandas
- seaborn
- matplotlib

## Importing and Exploring the data

The data is imported from the books.csv file. It is then explored using the following methods:

```python3
df.isnull().sum()
df.info()
df.describe()
```

## Data Exploration

The following visualizations are generated:

- Bar chart of top 10 books with the highest average rating
- Bar chart of top 10 authors with the most books
- Bar chart of top 10 books with the highest rating counts
- Distribution plot of average rating for all books
- Scatterplot of the relation between average rating and rating count
- Scatterplot of the relation between average rating and number of pages

## Data Preparation

The data is prepared for the recommendation system using the following steps:

The feature matrix is constructed by one-hot encoding the rating, language, and adding average rating and rating count columns

The feature matrix is normalized using MinMaxScaler
Building the Book Recommendation System

A nearest neighbors model is built using the feature matrix. The bookRecom function is used to recommend books based on user input.

## Link

Google colab file is located at https://colab.research.google.com/drive/1PN64dRGQOQ3OQQwyfcUAxmIjSp78_qvW
