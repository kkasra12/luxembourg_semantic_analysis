# luxembourg_semantic_analysis-
We will collect data from twitter and reddit about luxembourg post and we will fine-tune several mdoels on them

There are two folders `data` and `models`. it's better to include the codes to collect the data in each corresponding folder.

> IMPORTANT NOTE: **DON'T** put your api keys in the code!

To run the codes, install `requirements.txt`.
Copy and past your api keys in the `.env.dist` file and rename it to `.env`.

TODO: make better output

TODO: label the data

TODO: implement fasttext and a link that can be found at the end of the `models/sentiment_models.py` file

TODO: fix the 'add feedback' 

TODO: make the `show_data_<social_media>.html` file public for every social media. pass the `social_media` as an argument to the html file. and change the `show_data._twitter.html` file to `page_not_available.html` file. and call it when the `data.data_downloader` is raising `NotImplementedError` exception.

