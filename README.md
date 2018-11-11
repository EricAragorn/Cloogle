# Cloogo

## Inspiration
Shopping online clothing sites can be frustrating. On one hand, coming up with a specific group of right words to describe all the features of a piece of cloth is somewhat a effort-consuming guesswork, and the result often does not match the users' intent.
On the other hand, the existing visual search engine omits the flexibility for customization- users can only search the photo as a whole but cannot make any adjustments on a specific feature of the apparel. 

## What it does
Cloogle is a visual search engine for clothing sites to improve the flexibility, accuracy, and speed of the searching experience. 
On the consumer's side, Cloogle ditches the text box, and instead supports a more intuitive way of expression by allowing customers to picturize the style without converting the features into words. Comparing to the existing visual search, Cloogle does not search the provided picture as a whole, but first translates the picture into features describing the apparel, and after that enabling the users to customize the features freely as they want, and then search with the new features.
On the company's side, Cloogle not only helps the companies to reduce the time and effort in classifying items manually, and also help them better understand their customers, enabling them to make more accurate and customized recommendations.
By integrating Cloogle, retailers can reduce their operational cost, react faster to trends, better understand their customers, and provide the users with a more satisfactory shopping experience.

## How I built it
We first start collecting data using web crawler on different major online clothing platform (forever21, HM). Then, we use ResNet50 with global average pooling and multi-resolution features to fit 44 characteristics of 5 categories to train a neural network model to perform feature classification. On some of the features, we have achieved as high as 80 percent of accuracy. After that, with a rendering algorithm, features are used to construct a stylistic sketch representation of the clothes in the input image. We built on an HTML page and passed the input data upon a local server based on Django. Users could also freely adjust those features if necessary. The finalized features are then used to make customized recommendations.  

## Challenges I ran into
Crawling, data parsing, server building, UI designing, API connections

## Accomplishments that I'm proud of
We have finished data collection, training, and data visualization, 3 major tasks all within the time limit of 24h. This wouldn't be possible with the effort from every team member and seamless collaborations between us.  

## What I learned
Collaboration, google cloud ML engine.


## What's next for Cloogle
We will continue to improve the accuracy for pattern recognization and to polish the details of the sketch representation of features. Also, we will scale up our models to include more features that presumably will make the recommendations more accurate.

