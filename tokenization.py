# ! pip install nltk
import nltk
# nltk.download('punkt')
nltk.download('punkt', download_dir='C:/nltk_data')

from nltk.tokenize import sent_tokenize
# sent_tokenize : package to convert para to sentense

corpus="""The error message means that the data you're passing to your RandomForestClassifier ! for prediction has 4 features, but the model expects 5 features.!!!"""


print(sent_tokenize(corpus))

