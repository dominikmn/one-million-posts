import requests, uuid, json
from utils import loading, feature_engineering
from utils.config_azure import subscription_key, endpoint, location
import pandas as pd


def get_construction(subscription_key, endpoint, location):
    '''get construction to connect to microsoft azure translation api
    Arguments: subscription_key - the subscription key for azure
               endpoint - the url for the endpoint of the translation api
               location - the location for the api
    Return: constructed url - the url to connect to the api
            headers - the headers for the request to the api'''
    subscription_key = subscription_key
    endpoint = endpoint
    # Add your location, also known as region. The default is global.
    # This is required if using a Cognitive Services resource.
    location = location
    path = '/translate'
    constructed_url = endpoint + path

    constructed_url = endpoint + path
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    return constructed_url, headers


def translate_azure (df, col, lang, constructed_url, headers):
    '''translate a series from a dataset containing german texts to another language and back to german for data augmentation
    Arguments: df - a pandas dataframe
               col - the column name of the column containing german texts
               lang - the language for translation
               constructed url - the url to connect to the api
               headers - the headers for the request to the api
    Return: df2 - a dataset with the (back-)translated texts'''
    df2=df.copy()
    params = {
        'api-version': '3.0',
        'from': 'de',
        'to': lang
    }
    params2 = {
        'api-version': '3.0',
        'from': lang,
        'to': 'de'
    }
    df2['lang']=lang
    num_col = list(df2.columns).index(col)
    n=0
    error = 0
    for i in df[col]:
        try:
            body = [{'text': i}]
            request = requests.post(constructed_url, params=params, headers=headers, json=body)
            response = request.json()
            body2  =[{'text': response[0]['translations'][0]['text']}]

            request2 = requests.post(constructed_url, params=params2, headers=headers, json=body2)
            response = request2.json()
            df2.iloc[n, num_col]=response[0]['translations'][0]['text']
            
            if n%10==0:
                print(n)
        except:
            print(response)
            print()
            error+=1
        n+=1
    print(f'number of errors: {error}')
    return df2
    


def get_mult(df, label):
    '''helper function to determine how many translation iterations are necessary to get sufficent translations to augment the positive labels to 50%
    Arguments: df - a pandas dataframe containing the labeled data
               label - the label to be augmented
    Return: an int, how many translation iterations are neccesary'''
    pos = df[label].value_counts()[1]
    neg = df[label].value_counts()[0]
    return int(neg/pos)+1


def get_lang(mult):
    '''helper function to get a random list of languages for translations
    Arguments: mult - an int, how many translation iterations are neccesary
    Return: a list of languages for translation'''
    lang_list = pd.Series(['af', 'sq', 'am', 'ar', 'hy', 'as', 'az', 'bn', 
             'bs', 'bg', 'yue', 'ca', 'zh-Hans', 'zh-Hant', 
             'hr', 'cs', 'da', 'prs', 'nl', 'en', 'et', 'fj', 
             'fil', 'fi', 'fr', 'fr-ca', 'de', 'el', 'gu', 
             'ht', 'he', 'hi', 'mww', 'hu', 'is', 'id', 'iu', 
             'ga', 'it', 'ja', 'kn', 'kk', 'km', 'tlh-Latn', 
             'tlh-Piqd', 'ko', 'ku', 'kmr', 'lo', 'lv', 'lt', 
             'mg', 'ms', 'ml', 'mt', 'mi', 'mr', 'my', 'ne', 
             'nb', 'or', 'ps', 'fa', 'pl', 'pt', 'pt-pt', 
             'pa', 'otq', 'ro', 'ru', 'sm', 'sr-Cyrl', 'sr-Latn', 
             'sk', 'sl', 'es', 'sw', 'sv', 'ty', 'ta', 'te', 'th', 
             'ti', 'to', 'tr', 'uk', 'ur', 'vi', 'cy', 'yua'])
    return list(lang_list.sample(mult, random_state=42))


def get_trans_df(df, col, lang_list, constructed_url, headers):
    '''get a complete dataset with translations for augmentation
    Arguments: df - a pandas dataframe
               col - the column name of the column containing german texts
               lang_list - a list of languages for translation
               constructed url - the url to connect to the api
               headers - the headers for the request to the api
    Return: df2 - a complete dataset with the (back-)translated texts to augment the positive labels up to 50%'''
    df_temp = pd.DataFrame(columns=df.columns)
    for lang in lang_list:
        df_trans = translate_azure(df, col, lang, constructed_url, headers)
        df_temp=pd.concat((df_temp,df_trans))
    return df_temp


if __name__ == '__main__':
    df = loading.load_extended_posts(split='train')
    df.fillna(value={'headline':'', 'body':''}, inplace=True)
    df['text'] = df['headline']+" "+df['body']
    df['text']=df.text.str.replace('\n',' ').str.replace('\r', ' ')
    constructed_url, headers = get_construction(subscription_key, endpoint, location)
    label_list = [ 'label_argumentsused', 'label_discriminating', 'label_inappropriate', 'label_offtopic', 'label_personalstories',
                 'label_possiblyfeedback', 'label_sentimentnegative', 'label_sentiment_positive']
    try:
        for label in label_list:
            print(f'started {label}')
            mult = get_mult(df, label)
            lang_list = get_lang(mult)
            df_drop = df.dropna(subset=[label]).copy()
            df_pos = df_drop[df_drop[label]==1].copy()
            trans_df = get_trans_df(df_pos, 'text', lang_list, constructed_url, headers)
            trans_df.to_csv(f'./output/trans_{label}.csv')
            print(f'finished {label}')
    except:
        print('error occured')
