---
layout: post
title: Supreme Court opinion NLP project
---
# Supreme Court opinion NLP project

## Overview 
For our project, we conducted a sentiment analysis on the opinions of Supreme Court Justices with the aim to differentiate and highlight the unique "legal writing styles" of the Justices, which is beneficial for people learning about legal writing and may reveal Justices' legal philosophy. Our methodology included downloading large volumes of Supreme Court opinion PDFs from the official website. Then, we used OCR tools to detect and store the text before using regular expressions to separate the opinions and identify the author in order to construct our official dataset CSV. After preparing the data, we utilized NLP packages and tensorflow in order to find high prevalence words for each opinion type and author, as well as score the overall sentiment in the opinion. Once we created our models for both type and author classification based on the text, we tested these models on completely unseen data from the past 2 months. After examining our results, which were poor on the unseen data, we attempted to recreate our models after removing the justices from the training set who were not seen in the test set. As a result, our results seemed to improve.

![](https://github.com/jameswest25/IMDB_scraper/blob/main/IMDB_scraper/IMDB_scraper/Pictures/Screen%20Shot%202022-05-29%20at%203.46.59%20PM.png?raw=true)

## Technical Components
Web Scraping
OCR/Text Cleaning
Type and Author Classification
Retraining the Data and repeating Testing 

### Web Scraping
 	We began by scraping the Supreme Court Opinions Directory which contained pdf links of the Supreme Court opinions from 2021 to 2014. To create the scraper, we made a parse method that used the relevant css selectors and tags to acquire the opinion PDF links for each month of the year. Next we utilized a for loop to index through the list of PDF links and download the PDFs. A second parse method was created to go to the website links of each year and scrape and continue this process of downloading the PDFs. 

![](https://github.com/jameswest25/IMDB_scraper/blob/main/IMDB_scraper/IMDB_scraper/Pictures/Screen%20Shot%202022-05-29%20at%203.46.59%20PM.png?raw=true)

This is the spider for preliminary prints

```python
class courtscraper(scrapy.Spider):
    name = 'court_spider'
    
    start_urls = ['https://www.supremecourt.gov/opinions/USReports.aspx']

    def parse(self, response):

         pdfs = [a.attrib["href"] for a in response.css("div#accordion2 a")]
         prefix = "https://www.supremecourt.gov/opinions/"
         pdfs_urls = [prefix + suffix for suffix in pdfs]


         for url in pdfs_urls:
            item = Pic16BprojectItem() #define it items.py
            item['file_urls'] = [url]
            yield item
```
Code for the items.py file

```python
import scrapy

class Pic16BprojectItem(scrapy.Item):
    file_urls = scrapy.Field()
    files = scrapy.Field()
```

Code for the settings.py file needed in order to run the spider

```python


BOT_NAME = 'pic16bproject'

SPIDER_MODULES = ['pic16bproject.spiders']
NEWSPIDER_MODULE = 'pic16bproject.spiders'

ITEM_PIPELINES = {
    'scrapy.pipelines.files.FilesPipeline' : 1,
}

FILES_STORE = "pdf"
FILES_RESULT_FIELD = 'files'

ROBOTSTXT_OBEY = True
```

## OCR and Text Cleaning

Acknowledgement: https://www.geeksforgeeks.org/python-reading-contents-of-pdf-using-ocr-optical-character-recognition/


```python
# For every opinion PDF (donwloaded from spider)
for op in [i for i in os.listdir("./opinion_PDFs") if i[-3:] == 'pdf']:
    
    # *** Part 1 ***
    pages = convert_from_path("./opinion_PDFs/" + op, dpi = 300)
    image_counter = 1
    # Iterate through all the pages in this opinion and store as jpg
    for page in pages:
        # Declaring filename for each page of PDF as JPG
        # For each page, filename will be:
        # PDF page 1 -> page_1.jpg
        # ....
        # PDF page n -> page_n.jpg
        filename = "page_"+str(image_counter)+".jpg"
        # Save the image of the page in system
        page.save(filename, 'JPEG')
        # Increment the counter to update filename
        image_counter = image_counter + 1
    image_counter = image_counter - 1
    
    # *** Part 2 ***
    # Creating a text file to write the output
    outfile = "./opinion_txt/" + op.split(".")[0] + "_OCR.txt"
    # Open the file in append mode
    f = open(outfile, "w")
    
    # Iterate from 1 to total number of pages
    skipped_pages = []
    print("Starting OCR for " + re.findall('([0-9a-z-]+)_', op)[0])
    print("Reading page:")
    for i in range(1, image_counter + 1):
        print(str(i) + "...") if i==1 or i%10==0 or i==image_counter else None
        # Set filename to recognize text from
        filename = "page_" + str(i) + ".jpg"
        # Recognize the text as string in image using pytesserct
        text = pytesseract.image_to_string(Image.open(filename))
        # If the page is a syllabus page or not an opinion page
        # marked by "Opinion of the Court" or "Last_Name, J. dissenting/concurring"
        # skip and remove this file; no need to append text
        is_syllabus = re.search('Syllabus\n', text) is not None
        is_maj_op = re.search('Opinion of [A-Za-z., ]+\n', text) is not None
        is_dissent_concur_op = re.search('[A-Z]+, (C. )?J., (concurring|dissenting)?( in judgment)?', text) is not None
        if is_syllabus or ((not is_maj_op) and (not is_dissent_concur_op)):
            # note down the page was skipped, remove image, and move on to next page
            skipped_pages.append(i)
            os.remove(filename)
            continue
        # Restore sentences
        text = text.replace('-\n', '')
        # Roman numerals header
        text = re.sub('[\n]+[A-Za-z]{1,4}\n', '', text)
        # Remove headers
        text = re.sub("[\n]+SUPREME COURT OF THE UNITED STATES[\nA-Za-z0-9!'#%&()*+,-.\/\[\]:;<=>?@^_{|}~—’ ]+\[[A-Z][a-z]+ [0-9]+, [0-9]+\][\n]+",
                  ' ', text)
        text = re.sub('[^\n]((CHIEF )?JUSTICE ([A-Z]+)[-A-Za-z0-9 ,—\n]+)\.[* ]?[\n]{2}',
                  '!OP START!\\3!!!\\1!!!', text)
        text = re.sub('[\n]+', ' ', text) # Get rid of new lines and paragraphs
        text = re.sub('NOTICE: This opinion is subject to formal revision before publication in the preliminary print of the United States Reports. Readers are requested to noti[f]?y the Reporter of Decisions, Supreme Court of the United States, Washington, D.[ ]?C. [0-9]{5}, of any typographical or other formal errors, in order that corrections may be made before the preliminary print goes to press[\.]?',
                      '', text)
        text = re.sub('Cite as: [0-9]+[ ]?U.S.[_]* \([0-9]+\) ([0-9a-z ]+)?(Opinion of the Court )?([A-Z]+,( C.)? J., [a-z]+[ ]?)?',
                      '', text)
        text = re.sub(' JUSTICE [A-Z]+ took no part in the consideration or decision of this case[\.]?', '', text)
        text = re.sub('[0-9]+ [A-Z!&\'(),-.:; ]+ v. [A-Z!&\'(),-.:; ]+ (Opinion of the Court )?(dissenting[ ]?|concurring[ ]?)?',
                  '', text)
        # Remove * boundaries
        text = re.sub('([*][ ]?)+', '', text)
        # Eliminate "It is so ordered" after every majority opinion
        text = re.sub(' It is so ordered\. ', '', text)
        # Eliminate opinion header
        text = re.sub('Opinion of [A-Z]+, [C. ]?J[\.]?', '', text)
        # Separate opinions
        text = re.sub('!OP START!', '\n', text)
    
        # Write to text
        f.write(text)
    
        # After everything is done for the page, remove the page image
        os.remove(filename)
    # Close connection to .txt file after finishing writing
    f.close()
    
    # Now read in the newly created txt file as a pandas data frame if possible
    
    try:
        op_df = pd.read_csv("./opinion_txt/" + op.split(".")[0] + "_OCR.txt",
                            sep = re.escape("!!!"), engine = "python",
                            names = ["Author", "Header", "Text"])
        op_df.insert(1, "Docket_Number", re.findall("([-a-z0-9 ]+)_", op)[0])
        op_df["Type"] = op_df.Header.apply(opinion_classifier)
        
        # Lastly add all the opinion info to the main data frame
        opinion_df = opinion_df.append(op_df, ignore_index = True)
        os.remove("./opinion_PDFs/" + op)
        print("Task completed\nPages skipped: " + str(skipped_pages) + "\n")
    except:
        print("Error in CSV conversion. Pages NOT added!\n")
        
print("-----------------------\nAll assigned OCR Completed")
```

## Exploratory Data Analysis


```python
# Giving Colab access to drive (where csv files are stored)
from google.colab import drive
drive.mount('/content/gdrive')
```

    Mounted at /content/gdrive



```python
import pandas as pd # For data frame manipulation
!pip install afinn
from afinn import Afinn # for sentiment score
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting afinn
      Downloading afinn-0.1.tar.gz (52 kB)
    [K     |████████████████████████████████| 52 kB 1.1 MB/s 
    [?25hBuilding wheels for collected packages: afinn
      Building wheel for afinn (setup.py) ... [?25l[?25hdone
      Created wheel for afinn: filename=afinn-0.1-py3-none-any.whl size=53447 sha256=744d047e46cf248f31a705b6f4b171699bc649c54c3a7f077df381fd292d3f42
      Stored in directory: /root/.cache/pip/wheels/9d/16/3a/9f0953027434eab5dadf3f33ab3298fa95afa8292fcf7aba75
    Successfully built afinn
    Installing collected packages: afinn
    Successfully installed afinn-0.1



```python
# Load in the dataframe for train/test/validation
train = pd.read_csv('gdrive/My Drive/PIC16B-Final-Project/train_validate.csv',
                   usecols = ["Author", "Text", "Type"])
train.head(10)
```


```python
train.groupby(by = ["Author", "Type"]).size().reset_index().pivot(index = "Author", columns = "Type", values = 0)
```



### Type and Author Classification
	We used tensorflow in order to classify all of the opinion types and justices, labeled as authors, based on the text alone. To do this, we created two data frames: one with type and text as the columns, and another with author and text as the columns. Then, we converted each type and column into integer labels using a label encoder in order to move forward with our classification task. We split our data into 70% training, 10% validation, and 20% testing in order to train our models and compare our resulting accuracies,. Both the type and author models implemented a sequential model that used an embedding layer, two dropout layers, a 1D global average pooling layer, and a dense layer. The dimensions for the output and dense layer were altered based on the total number of opinion types (4) and total number of authors (12). We experienced great success with the training and validation accuracies for both models. For the type model, the training accuracies hovered around 92% and the validation accuracies settled around 99% as the later epochs were completed. For the author model, the training accuracies hovered around 87% and the validation accuracies settled around 97% as the later epochs were completed. Further, we did not worry too much about overfitting as there was a large amount of overlap between training and validation accuracies and there was never too much of a dropoff between the two. After training our models, we evaluated them on the testing set which was the random 20% of the original data. Once again, experienced great success as the type and author test accuracies were ~99.5% and ~95.6%, respectively. Thus, our models performed very well on the original dataset.
	However, we also created an additional testing set that included unseen data from the past two months alone. This is where our models seemed to falter. Specifically, our type model testing accuracy was ~28.1% and our author model testing accuracy was ~3.1%. These are clearly much lower than the testing accuracies from our initial data. Thus, we performed further evaluation of our datasets and noticed some variations. Specifically, the unseen test set which has all the data from the last two months, consisted of fewer authors than our original data. So, we removed the justices from the original dataset who were not seen in the data from the last two months and retrained and tested our models once again. Similar to the first time, the training and validation accuracies were very high. However, we did notice a slight increase in our testing accuracies as the type model improved to ~34.4% and our author model improved to ~15.6%. Although these are still rather low, we believe that further inspection into our test dataset would provide us with more clarity about potential improvements that we could make to our model so that it performs better with the testing data.

#### Type Classification based on Text using Tensorflow

```python
import numpy as np
import tensorflow as tf
import re
import string
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import losses

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA
import plotly.express as px 
```


```python
le = LabelEncoder()
train["Type"] = le.fit_transform(train["Type"]) # Recode Type to numeric labels
```


```python
# Encoding dictionary
type_dict = dict(zip(le.transform(le.classes_), le.classes_))
type_dict
```




    {0: 'Concurrence', 1: 'Concurrence in Judgment', 2: 'Dissent', 3: 'Opinion'}




```python
type_df = train.drop(["Author"], axis = 1)
```



```python
type_df.groupby("Type").size()
```




    Type
    0    154
    1    117
    2    347
    3    424
    dtype: int64




```python
type_train_df = tf.data.Dataset.from_tensor_slices((train["Text"], train["Type"]))
```


```python
type_train_df = type_train_df.shuffle(buffer_size = len(type_train_df))

# Split data into 70% train, 10% validation, 20% test
train_size = int(0.7*len(type_train_df)) 
val_size = int(0.1*len(type_train_df))

type_train = type_train_df.take(train_size) 
type_val = type_train_df.skip(train_size).take(val_size)
type_test = type_train_df.skip(train_size + val_size)
```


```python
# Length of the train, validation, and test data sets
len(type_train), len(type_val), len(type_test)
```




    (729, 104, 209)




```python
# standardize the text:
# remove punctuation, convert all to lowercase, remove numers
def standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    no_punctuation = tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation),'')
    standardized = tf.strings.regex_replace(no_punctuation, '[0-9]+', '*')
    return standardized
```


```python
max_tokens = 2000
sequence_length = 25 

vectorize_layer = TextVectorization(
    standardize =  standardization, 
    output_mode = 'int', 
    max_tokens = max_tokens, 
    output_sequence_length =  sequence_length
)

opinion_type = type_train.map(lambda x, y: x)
vectorize_layer.adapt(opinion_type)
```


```python
def vectorize_pred(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), [label]

train_vec = type_train.map(vectorize_pred)
val_vec = type_val.map(vectorize_pred)
test_vec = type_test.map(vectorize_pred)
```


```python
type_model = tf.keras.Sequential([
    layers.Embedding(max_tokens, output_dim = 4, name = "embedding"),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(), 
    layers.Dropout(0.2), 
    layers.Dense(4)
])
```


```python
type_model.compile(loss = losses.SparseCategoricalCrossentropy(from_logits = True),
                   optimizer = "adam", 
                   metrics = ["accuracy"])
```


```python
type_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       (None, None, 4)           8000      
                                                                     
     dropout (Dropout)           (None, None, 4)           0         
                                                                     
     global_average_pooling1d (G  (None, 4)                0         
     lobalAveragePooling1D)                                          
                                                                     
     dropout_1 (Dropout)         (None, 4)                 0         
                                                                     
     dense (Dense)               (None, 4)                 20        
                                                                     
    =================================================================
    Total params: 8,020
    Trainable params: 8,020
    Non-trainable params: 0
    _________________________________________________________________



```python
history = type_model.fit(train_vec, epochs = 80, validation_data = val_vec)
```
    Epoch 80/80
    729/729 [==============================] - 3s 4ms/step - loss: 0.2210 - accuracy: 0.9287 - val_loss: 0.0988 - val_accuracy: 0.9808



```python
def plot_model(history):
    model_history = pd.DataFrame(history.history)
    model_history['epoch'] = history.epoch
    fig, ax = plt.subplots(1, figsize=(8,6))
    num_epochs = model_history.shape[0]
    ax.plot(np.arange(0, num_epochs), model_history["accuracy"], 
        label="Training Accuracy")
    ax.plot(np.arange(0, num_epochs), model_history["val_accuracy"], 
        label="Validation Accuracy")
    ax.legend()
    plt.tight_layout()
    plt.show()
```


```python
plot_model(history)
```


    
![png](PIC16B_SCOTUS_Project_Presentation_files/PIC16B_SCOTUS_Project_Presentation_30_0.png)
    



```python
type_model.evaluate(test_vec) # Checking accuracy on the test data set
```

    209/209 [==============================] - 1s 3ms/step - loss: 0.0984 - accuracy: 0.9952





    [0.09842098504304886, 0.9952152967453003]




```python
weights = type_model.get_layer('embedding').get_weights()[0] 
vocab = vectorize_layer.get_vocabulary()                

pca = PCA(n_components = 2)
weights = pca.fit_transform(weights)
```


```python
print(pca.explained_variance_ratio_) # Percentage of variance explained by each PC
np.sum(pca.explained_variance_ratio_) # total percentage explained by our PCs
```

    [0.550473  0.3038923]





    0.8543653




```python
embedding_df = pd.DataFrame({
    'word' : vocab, 
    'x0'   : weights[:,0],
    'x1'   : weights[:,1]
})

fig = px.scatter(embedding_df, 
                 x = "x0", 
                 y = "x1", 
                 size = [2]*len(embedding_df),
                 hover_name = "word")

fig.show()
```



#### Author Classification based on Text using Tensorflow


```python
train["Author"] = le.fit_transform(train["Author"])
author_dict = dict(zip(le.transform(le.classes_), le.classes_))
author_dict
```




    {0: 'ALITO',
     1: 'BARRETT',
     2: 'BREYER',
     3: 'GINSBURG',
     4: 'GORSUCH',
     5: 'KAGAN',
     6: 'KAVANAUGH',
     7: 'KENNEDY',
     8: 'ROBERTS',
     9: 'SCALIA',
     10: 'SOTOMAYOR',
     11: 'THOMAS'}




```python
author_df = train.drop(["Type"], axis = 1)
author_df
```


```python
author_train_df = tf.data.Dataset.from_tensor_slices((author_df["Text"], author_df["Author"]))
author_train_df = author_train_df.shuffle(buffer_size = len(author_train_df))

# Split data into 70% train, 10% validation, 20% test
train_size = int(0.7*len(author_train_df)) 
val_size = int(0.1*len(author_train_df))

author_train = author_train_df.take(train_size) 
author_val = author_train_df.skip(train_size).take(val_size)
author_test = author_train_df.skip(train_size + val_size)

opinion_author = author_train.map(lambda x, y: x)
vectorize_layer.adapt(opinion_author)

author_train_vec = author_train.map(vectorize_pred)
author_val_vec = author_val.map(vectorize_pred)
author_test_vec = author_test.map(vectorize_pred)

author_model = tf.keras.Sequential([
    layers.Embedding(max_tokens, output_dim = 12, name = "embedding"),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(), 
    layers.Dropout(0.2), 
    layers.Dense(12)
])

author_model.compile(loss = losses.SparseCategoricalCrossentropy(from_logits = True),
                   optimizer = "adam",
                   metrics = ["accuracy"])

author_model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       (None, None, 12)          24000     
                                                                     
     dropout_2 (Dropout)         (None, None, 12)          0         
                                                                     
     global_average_pooling1d_1   (None, 12)               0         
     (GlobalAveragePooling1D)                                        
                                                                     
     dropout_3 (Dropout)         (None, 12)                0         
                                                                     
     dense_1 (Dense)             (None, 12)                156       
                                                                     
    =================================================================
    Total params: 24,156
    Trainable params: 24,156
    Non-trainable params: 0
    _________________________________________________________________



```python
history1 = author_model.fit(author_train_vec, epochs = 50, validation_data = author_val_vec)
```

    Epoch 50/50
    729/729 [==============================] - 3s 4ms/step - loss: 0.5179 - accuracy: 0.8875 - val_loss: 0.3543 - val_accuracy: 0.9808



```python
plot_model(history1)
```


    
![png](PIC16B_SCOTUS_Project_Presentation_files/PIC16B_SCOTUS_Project_Presentation_40_0.png)
    



```python
author_model.evaluate(author_test_vec) # Checking accuracy on test data set
```

    209/209 [==============================] - 1s 3ms/step - loss: 0.3657 - accuracy: 0.9569





    [0.3656679093837738, 0.9569377899169922]




```python
weights = author_model.get_layer('embedding').get_weights()[0] 
vocab = vectorize_layer.get_vocabulary()                

pca = PCA(n_components = 8)
weights = pca.fit_transform(weights)
```


```python
# Percentage of variances explained
print(pca.explained_variance_ratio_)
np.sum(pca.explained_variance_ratio_)
```

    [0.26117828 0.17431074 0.1260937  0.11706422 0.1015166  0.07762694
     0.049953   0.04238212]





    0.9501256




```python
embedding_df = pd.DataFrame({
    'word' : vocab, 
    'x0'   : weights[:,0],
    'x1'   : weights[:,1]
})

fig = px.scatter(embedding_df, 
                 x = "x0", 
                 y = "x1", 
                 size = [2]*len(embedding_df),
                 hover_name = "word")

fig.show()
```

#### Testing Models on Completely Unseen Data


```python
test = pd.read_csv('gdrive/My Drive/PIC16B-Final-Project/test.csv',
                   usecols = ["Author", "Text", "Type", "Author_Type"])
```


```python
# True labels
test["Type"] = le.fit_transform(test["Type"])
true_type = [type_dict[i] for i in test.Type.to_numpy()]
true_type_label = test.Type.to_numpy()
```


```python
type_test_df  = tf.data.Dataset.from_tensor_slices((test["Text"], test["Type"]))
type_test_vec = type_test_df.map(vectorize_pred)

# Predicted labels
pred_type_label = type_model.predict(type_test_vec).argmax(axis = 1)
predicted_type = [type_dict[i] for i in pred_type_label]
```


```python
correct_type = (true_type_label == pred_type_label)
pd.DataFrame(list(zip(predicted_type, true_type, correct_type)),
             columns = ["Predicted", "Actual", "Correct"])
```


```python
type_model.evaluate(type_test_vec)
```

    32/32 [==============================] - 0s 3ms/step - loss: 2.1633 - accuracy: 0.2812





    [2.16332745552063, 0.28125]




```python
test["Author"] = le.fit_transform(test["Author"])
# True Labels
true_author = [author_dict[i] for i in test.Author.to_numpy()]
true_author_label = test.Author.to_numpy()
```


```python
author_test_df = tf.data.Dataset.from_tensor_slices((test["Text"], test["Author"]))
author_test_vec = author_test_df.map(vectorize_pred)
# Predicted labels
predicted_author = [author_dict[i] for i in author_model.predict(author_test_vec).argmax(axis = 1)]
pred_author = author_model.predict(author_test_vec).argmax(axis = 1)
```


```python
correct = true_author_label == pred_author
pd.DataFrame(list(zip(predicted_author, true_author, correct)),
             columns = ["Predicted", "Actual", "Correct"])
```




```python
author_model.evaluate(author_test_vec)
```

    32/32 [==============================] - 0s 3ms/step - loss: 3.6711 - accuracy: 0.0312





    [3.6711015701293945, 0.03125]



#### Differences in Test and Train --> Could be reason why test acc is so low


```python
tv_data = pd.read_csv('gdrive/My Drive/PIC16B-Final-Project/train_validate.csv',
                   usecols = ["Author", "Text", "Type"])
te_data = pd.read_csv('gdrive/My Drive/PIC16B-Final-Project/test.csv',
                   usecols = ["Author", "Text", "Type"])
```


```python
tv_summary = tv_data.groupby(by = ["Author", "Type"]).size().reset_index().pivot(index = "Author", columns = "Type", values = 0)
tv_summary["Total"] = tv_summary.sum(axis = 1)
tv_summary
```



```python
te_summary = te_data.groupby(by = ["Author", "Type"]).size().reset_index().pivot(index = "Author", columns = "Type", values = 0).fillna(0).astype('int')
te_summary["Total"] = te_summary.sum(axis = 1)
te_summary
```



#### Removing Justices who are not present in the Test data set

##### Type Classification


```python
# Re-load train and test data sets
train = pd.read_csv('gdrive/My Drive/PIC16B-Final-Project/train_validate.csv',
                   usecols = ["Author", "Text", "Type"])
train_lmt = train[~train.Author.isin(["GINSBURG", "KENNEDY", "SCALIA"])].reset_index().drop("index", axis = 1)
train_lmt.head(10)
```



```python
le = LabelEncoder()
train_lmt["Type"] = le.fit_transform(train_lmt["Type"])

type_df = train_lmt.drop(["Author"], axis = 1)
type_df

type_train_df = tf.data.Dataset.from_tensor_slices((train_lmt["Text"], train_lmt["Type"]))

type_train_df = type_train_df.shuffle(buffer_size = len(type_train_df))

# Split data into 70% train, 10% validation, 20% test
train_size = int(0.7*len(type_train_df)) 
val_size = int(0.1*len(type_train_df))

type_train = type_train_df.take(train_size) 
type_val = type_train_df.skip(train_size).take(val_size)
type_test = type_train_df.skip(train_size + val_size)

opinion_type = type_train.map(lambda x, y: x)
vectorize_layer.adapt(opinion_type)

train_vec = type_train.map(vectorize_pred)
val_vec = type_val.map(vectorize_pred)
test_vec = type_test.map(vectorize_pred)

type_model.compile(loss = losses.SparseCategoricalCrossentropy(from_logits = True),
                   optimizer = "adam", 
                   metrics = ["accuracy"])

type_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       (None, None, 4)           8000      
                                                                     
     dropout (Dropout)           (None, None, 4)           0         
                                                                     
     global_average_pooling1d (G  (None, 4)                0         
     lobalAveragePooling1D)                                          
                                                                     
     dropout_1 (Dropout)         (None, 4)                 0         
                                                                     
     dense (Dense)               (None, 4)                 20        
                                                                     
    =================================================================
    Total params: 8,020
    Trainable params: 8,020
    Non-trainable params: 0
    _________________________________________________________________



```python
history3 = type_model.fit(train_vec, epochs = 80, validation_data = val_vec)
plot_model(history3)
```

    Epoch 80/80
    616/616 [==============================] - 3s 5ms/step - loss: 0.2414 - accuracy: 0.9172 - val_loss: 0.0878 - val_accuracy: 0.9886



    
![png](PIC16B_SCOTUS_Project_Presentation_files/PIC16B_SCOTUS_Project_Presentation_63_1.png)
    



```python
type_model.evaluate(test_vec) # 0.9886
```

    176/176 [==============================] - 1s 3ms/step - loss: 0.0951 - accuracy: 0.9886





    [0.09507766366004944, 0.9886363744735718]




```python
weights = type_model.get_layer('embedding').get_weights()[0] 
vocab = vectorize_layer.get_vocabulary()                

pca = PCA(n_components = 3)
weights = pca.fit_transform(weights)
```


```python
print(pca.explained_variance_ratio_)
np.sum(pca.explained_variance_ratio_)
```

    [0.5351038  0.29282776 0.16341491]





    0.9913464




```python
embedding_df = pd.DataFrame({
    'word' : vocab, 
    'x0'   : weights[:,0],
    'x1'   : weights[:,1]
})

fig = px.scatter(embedding_df, 
                 x = "x0", 
                 y = "x1", 
                 size = [2]*len(embedding_df),
                 hover_name = "word")

fig.show()
```


##### Author Classification


```python
train = pd.read_csv('gdrive/My Drive/PIC16B-Final-Project/train_validate.csv',
                   usecols = ["Author", "Text", "Type"])
train_lmt = train[~train.Author.isin(["GINSBURG", "KENNEDY", "SCALIA"])].reset_index().drop("index", axis = 1)
train_lmt["Author"] = le.fit_transform(train_lmt["Author"])
author_dict = dict(zip(le.transform(le.classes_), le.classes_))
author_dict
```




    {0: 'ALITO',
     1: 'BARRETT',
     2: 'BREYER',
     3: 'GORSUCH',
     4: 'KAGAN',
     5: 'KAVANAUGH',
     6: 'ROBERTS',
     7: 'SOTOMAYOR',
     8: 'THOMAS'}




```python
author_df = train_lmt.drop(["Type"], axis = 1)
author_train_df = tf.data.Dataset.from_tensor_slices((author_df["Text"], author_df["Author"]))
author_train_df = author_train_df.shuffle(buffer_size = len(author_train_df))

# Split data into 70% train, 10% validation, 20% test
train_size = int(0.7*len(author_train_df)) 
val_size = int(0.1*len(author_train_df))

author_train = author_train_df.take(train_size) 
author_val = author_train_df.skip(train_size).take(val_size)
author_test = author_train_df.skip(train_size + val_size)

opinion_author = author_train.map(lambda x, y: x)
vectorize_layer.adapt(opinion_author)

author_train_vec = author_train.map(vectorize_pred)
author_val_vec = author_val.map(vectorize_pred)
author_test_vec = author_test.map(vectorize_pred)

author_model = tf.keras.Sequential([
    layers.Embedding(max_tokens, output_dim = 12, name = "embedding"),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(), 
    layers.Dropout(0.2), 
    layers.Dense(12)
])

author_model.compile(loss = losses.SparseCategoricalCrossentropy(from_logits = True),
                   optimizer = "adam",
                   metrics = ["accuracy"])

author_model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       (None, None, 12)          24000     
                                                                     
     dropout_4 (Dropout)         (None, None, 12)          0         
                                                                     
     global_average_pooling1d_2   (None, 12)               0         
     (GlobalAveragePooling1D)                                        
                                                                     
     dropout_5 (Dropout)         (None, 12)                0         
                                                                     
     dense_2 (Dense)             (None, 12)                156       
                                                                     
    =================================================================
    Total params: 24,156
    Trainable params: 24,156
    Non-trainable params: 0
    _________________________________________________________________



```python
history4 = author_model.fit(author_train_vec, epochs = 50, validation_data = author_val_vec)
```

    Epoch 50/50
    616/616 [==============================] - 3s 4ms/step - loss: 0.4533 - accuracy: 0.9010 - val_loss: 0.3121 - val_accuracy: 0.9659



```python
plot_model(history4)
```


    
![png](PIC16B_SCOTUS_Project_Presentation_files/PIC16B_SCOTUS_Project_Presentation_72_0.png)
    



```python
author_model.evaluate(author_test_vec) # 0.9716 with 50 epochs
```

    176/176 [==============================] - 1s 3ms/step - loss: 0.2943 - accuracy: 0.9886





    [0.2943049371242523, 0.9886363744735718]




```python
weights = author_model.get_layer('embedding').get_weights()[0] 
vocab = vectorize_layer.get_vocabulary()                

pca = PCA(n_components = 7)
weights = pca.fit_transform(weights)
```


```python
print(pca.explained_variance_ratio_)
np.sum(pca.explained_variance_ratio_)
```

    [0.29698044 0.23482043 0.171238   0.11807768 0.07483307 0.03601614
     0.03262113]





    0.9645869




```python
embedding_df = pd.DataFrame({
    'word' : vocab, 
    'x0'   : weights[:,0],
    'x1'   : weights[:,1]
})

fig = px.scatter(embedding_df, 
                 x = "x0", 
                 y = "x1", 
                 size = [2]*len(embedding_df),
                 hover_name = "word")

fig.show()
```

##### Testing Models on Completely Unseen Data


```python
test = pd.read_csv('gdrive/My Drive/PIC16B-Final-Project/test.csv',
                   usecols = ["Author", "Text", "Type", "Author_Type"])
```


```python
# True labels
test["Type"] = le.fit_transform(test["Type"])
true_type = [type_dict[i] for i in test.Type.to_numpy()]
true_type_label = test.Type.to_numpy()
true_type_label
```




    array([3, 2, 3, 0, 1, 1, 3, 2, 3, 3, 0, 2, 3, 3, 2, 3, 3, 2, 3, 3, 0, 0,
           2, 3, 3, 0, 2, 2, 3, 2, 3, 2])




```python
type_test_df  = tf.data.Dataset.from_tensor_slices((test["Text"], test["Type"]))
type_test_vec = type_test_df.map(vectorize_pred)

# Predicted labels
pred_type_label = type_model.predict(type_test_vec).argmax(axis = 1)
predicted_type = [type_dict[i] for i in pred_type_label]
pred_type_label
```




    array([3, 3, 2, 2, 2, 3, 2, 2, 3, 3, 3, 2, 2, 2, 3, 2, 3, 1, 3, 3, 3, 3,
           3, 2, 2, 0, 0, 2, 2, 3, 2, 2])




```python
correct_type = (true_type_label == pred_type_label)
pd.DataFrame(list(zip(predicted_type, true_type, correct_type)),
             columns = ["Predicted", "Actual", "Correct"])
```

```python
type_model.evaluate(type_test_vec)
```

    32/32 [==============================] - 0s 3ms/step - loss: 2.1762 - accuracy: 0.3438





    [2.176164150238037, 0.34375]




```python
test["Author"] = le.fit_transform(test["Author"])
# True Labels
true_author = [author_dict[i] for i in test.Author.to_numpy()]
true_author_label = test.Author.to_numpy()
```


```python
author_test_df = tf.data.Dataset.from_tensor_slices((test["Text"], test["Author"]))
author_test_vec = author_test_df.map(vectorize_pred)
# Predicted labels
predicted_author = [author_dict[i] for i in author_model.predict(author_test_vec).argmax(axis = 1)]
pred_author = author_model.predict(author_test_vec).argmax(axis = 1)
pred_author
```




    array([8, 0, 3, 8, 8, 0, 7, 2, 8, 6, 8, 0, 8, 6, 6, 3, 2, 2, 7, 8, 8, 7,
           2, 8, 0, 7, 8, 2, 7, 0, 2, 0])




```python
correct = true_author_label == pred_author
pd.DataFrame(list(zip(predicted_author, true_author, correct)),
             columns = ["Predicted", "Actual", "Correct"])
```


```python
author_model.evaluate(author_test_vec)
```

    32/32 [==============================] - 0s 3ms/step - loss: 2.6749 - accuracy: 0.1562





    [2.6749370098114014, 0.15625]


