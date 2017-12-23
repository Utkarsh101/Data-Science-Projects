

class Tweet:
    def __init__(self):
        pass

class TweetCriteria:
    def __init__(self):
        self.maxTweets = 0

    def setUsername(self, username):
        self.username = username
        return self

    def setSince(self, since):
        self.since = since
        return self

    def setUntil(self, until):
        self.until = until
        return self

    def setQuerySearch(self, querySearch):
        self.querySearch = querySearch
        return self

    def setMaxTweets(self, maxTweets):
        self.maxTweets = maxTweets
        return self


import urllib, json, re, datetime
import urllib.parse
import urllib.request
import codecs
from pyquery import PyQuery


class TweetManager:
    def __init__(self):
        pass

    @staticmethod
    def getTweets(tweetCriteria):

        refreshCursor = ''

        results = []

        check = True

        while True:
            json = TweetManager.getJsonReponse(tweetCriteria, refreshCursor)
            if len(json['items_html'].strip()) == 0:
                break

            refreshCursor = json['min_position']
            tweets = PyQuery(json['items_html'])('div.js-stream-tweet')

            if len(tweets) == 0:
                break

            for tweetHTML in tweets:
                tweetPQ = PyQuery(tweetHTML)
                tweet = Tweet()

                if check:
                    #print(tweetPQ.html().encode("utf-8"))
                    check = False

                usernameTweet = tweetPQ("a.js-user-profile-link").attr('href');
                txt = re.sub(r"\s+", " ",
                             re.sub(r"[^\x00-\x7F]", "", tweetPQ("p.js-tweet-text").text()).replace('# ', '#').replace(
                                 '@ ', '@'));
                retweets = int(tweetPQ("span.ProfileTweet-action--retweet span.ProfileTweet-actionCount").attr(
                    "data-tweet-stat-count").replace(",", ""));
                favorites = int(tweetPQ("span.ProfileTweet-action--favorite span.ProfileTweet-actionCount").attr(
                    "data-tweet-stat-count").replace(",", ""));
                dateSec = int(tweetPQ("small.time span.js-short-timestamp").attr("data-time"));
                id = tweetPQ.attr("data-tweet-id");
                permalink = tweetPQ.attr("data-permalink-path");

                geo = ''
                geoSpan = tweetPQ('span.Tweet-geo')
                if len(geoSpan) > 0:
                    geo = geoSpan.attr('title')

                tweet.id = id
                tweet.permalink = 'https://twitter.com' + permalink
                tweet.username = usernameTweet.strip('/');
                tweet.text = txt
                tweet.date = datetime.datetime.fromtimestamp(dateSec)
                tweet.retweets = retweets
                tweet.favorites = favorites
                tweet.mentions = " ".join(re.compile('(@\\w*)').findall(tweet.text))
                tweet.hashtags = " ".join(re.compile('(#\\w*)').findall(tweet.text))
                tweet.geo = geo

                results.append(tweet)

                if tweetCriteria.maxTweets > 0 and len(results) >= tweetCriteria.maxTweets:
                    return results

        return results

    @staticmethod
    def getJsonReponse(tweetCriteria, refreshCursor):
        url = "https://twitter.com/i/search/timeline?f=realtime&q=%s&src=typd&max_position=%s"

        urlGetData = ''
        if hasattr(tweetCriteria, 'username'):
            urlGetData += ' from:' + tweetCriteria.username

        if hasattr(tweetCriteria, 'since'):
            urlGetData += ' since:' + tweetCriteria.since

        if hasattr(tweetCriteria, 'until'):
            urlGetData += ' until:' + tweetCriteria.until

        if hasattr(tweetCriteria, 'querySearch'):
            urlGetData += ' ' + tweetCriteria.querySearch

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'}

        url = url % (urllib.parse.quote(urlGetData), refreshCursor)

        req = urllib.request.Request(url, headers=headers)

        jsonResponse = urllib.request.urlopen(req).read().decode('utf8')

        dataJson = json.loads(jsonResponse)

        return dataJson

import sys, getopt, datetime

def main(argv):
    if len(argv) == 0:
        print('You must pass some parameters. Use \"-h\" to help.')
        return

    if len(argv) == 1 and argv[0] == '-h':
        print("""\nTo use this jar, you can pass the folowing attributes:
    username: Username of a specific twitter account (without @)
       since: The lower bound date (yyyy-mm-aa)
       until: The upper bound date (yyyy-mm-aa)
 querysearch: A query text to be matched
   maxtweets: The maximum number of tweets to retrieve

 \nExamples:
 # Example 1 - Get tweets by username [barackobama]
 python Export.py --username 'barackobama' --maxtweets 1\n

 # Example 2 - Get tweets by query search [europe refugees]
 python Export.py --querysearch 'europe refugees' --maxtweets 1\n""")
        return

    try:
        opts, args = getopt.getopt(argv, "", ("username=", "since=", "until=", "querysearch=", "maxtweets="))

        tweetCriteria = TweetCriteria()

        #print('max tweets : ' + int(args[1]))
        for opt, arg in opts:
            if opt == '--username':
                tweetCriteria.username = arg

            elif opt == '--since':
                tweetCriteria.since = arg

            elif opt == '--until':
                tweetCriteria.until = arg

            elif opt == '--querysearch':
                tweetCriteria.querySearch = arg

            elif opt == '--maxtweets':
                #print('max tweets : ' + int(arg))
                tweetCriteria.maxTweets = int(arg)

        import csv

        outputFile = open("Tweets_Nike.csv", "w+")

        print ('Searching for tweets...\n')

        fieldnames = ['TweetID', 'Username', 'Date', 'Retweets', 'Favorites', 'Text', 'Mentions', 'Permalink']
        writer = csv.DictWriter(outputFile, fieldnames=fieldnames)
        writer.writeheader()

        for t in TweetManager.getTweets(tweetCriteria):

            writer.writerow(
                {'TweetID': t.id,
                 'Username': t.username,
                 'Date': t.date.strftime("%Y-%m-%d %H:%M"),
                 'Retweets': t.retweets,
                 'Favorites': t.favorites,
                 'Text': t.text,
                 'Mentions': t.mentions,
                 'Permalink': t.permalink
                 })

        outputFile.close()

        print ('Tweets successfully extracted.')

    except arg:
        print ('Arguments parser error, try -h' + arg)


if __name__ == '__main__':
    main(sys.argv[1:])
