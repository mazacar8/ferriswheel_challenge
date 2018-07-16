# Ferriswheel Challenge

This code categorizes the 1000 fashion items described in product_data.json into 13 possible categories specified in product_categories.txt. It does so by comparing the Word2Vec embeddings of words in the descriptions of these items to the category names and their synonyms. The most likely category is then chosen given the similarity between the description and category is greater than a certain threshold. If less than the threshold, the item is classified into the "others" category.

## Requirements:
 
The only required files are the data JSON, product_categories.txt and the Word2Vec Model which can be downloaded at https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit. Can optionally include a stopwords file for words to discard during classification and a synonyms file for the given categories. The first word of each new line in this latter file must be the name of a category and must contain all categories except "others".

## Usage:

```
ferriswheel_challenge.py [-h] [--synonyms SYNONYMS_FILENAME]
                                [--stop STOPWORDS_FILENAME]
                                [--images WRITE_IMAGES]
                                data_filename categories_filename w2v_filename
```

**Argument Parser for Product Classifier**

positional arguments:
data_filename         Path to json product data file\
categories_filename   Path to txt file containing list of categories
w2v_filename          Path to Word2Vec Embeddings File

optional arguments:
  -h, --help            show this help message and exit
  --synonyms SYNONYMS_FILENAME
                        Path to txt file containing list of synonyms for
                        categories. First synonym MUST be category name.
  --stop STOPWORDS_FILENAME
                        Optional path to txt file containing stopwords
  --images WRITE_IMAGES
                        0 or 1. Option to write images of items into assigned
                        category folders

## Output:

The code creates a file "categorized_products.json" that includes the same information as "product_data.json" but with an additional field "category".

If --images is set to 1, this code also creates a folder for each category with images of the prodcuts classified into that category.
