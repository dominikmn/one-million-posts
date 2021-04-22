from string import punctuation
import re

def strip_punct(series):
    '''Strip puncation of series of strings
    Arguments: series - a series containing strings
    Return: new-series - a series of strings without punctuation'''
    new_series = series.str.replace(r'[^\w\s]+', '', regex=True)
    return new_series

def series_apply_chaining(series, functions):
    for f in functions:
        series = series.apply(f)
    return series

def normalize(txt, url_emoji_dummy=False):
    """


    """
    txt = txt.lower()

    url_dummy = ' '
    emoji_dummy = ' '
    if url_emoji_dummy:
        url_dummy = 'URL'
        emoji_dummy = 'EMOJI'
    # replace URLs
    # URLs starting with http(s) or ftp(s)
    url_re1 = re.compile(r'(?:ftp|http)s?://[\w\d:#@%/;$()~_?+=\,.&#!|-]+')
    txt = url_re1.sub(url_dummy, txt)
    # URLs starting with www.example.com
    url_re2 = re.compile(r'\bwww\.[a-zA-Z0-9-]{2,63}\.[\w\d:#@%/;$()~_?+=\,.&#!|-]+')
    txt = url_re2.sub(url_dummy, txt)
    # URLs short version example.com 
    url_re3 = re.compile(r'\b[a-zA-Z0-9.]+\.(?:com|org|net|io)')
    txt = url_re3.sub(url_dummy, txt)

    # replace emoticons
    # "Western" emoticons such as =-D and (^:
    s = r"(?:^|(?<=[\s:]))"      # beginning or whitespace required before
    s += r"(?:"                  # begin emoticon
    s += r"(?:"                  # begin "forward" emoticons like :-)
    s += r"[<>]?"                # optinal hat/brow
    s += r"[:;=8xX]"             # eyes
    s += r"[o*'^-]?"             # optional nose
    s += r"[(){}[\]dDpP/\\|@3]+" # mouth
    s += r")"                    # end "forward" emoticons
    s += r"|"                    # or
    s += r"(?:"                  # begin "backward" emoticons like (-:
    s += r"[(){}[\]dDpP/\\|@3]+" # mouth
    s += r"[o*'^-]?"             # optional nose
    s += r"[:;=8xX]"             # eyes
    s += r"[<>]?"                # optinal hat/brow
    s += r")"                    # end "backward" emoticons
    # "Eastern" emoticons like ^^ and o_O
    s += r"|"                    # or
    s += r"(?:\^\^)|"            # 'eastern' emoji
    s += r"(?:[<(]?o_o[)>]?)"    # 'eastern' emoji. Has to be lower-case 'o' because of the preceeding .lower() further above
    s += r")"                    # end emoticon
    s += r"(?=\s|$)"             # white space or end required after
    emoticon_re = re.compile(s)
    txt = emoticon_re.sub(emoji_dummy, txt)  #replace with 'EMOTICON but keep preceeding and trailing space/linefeed

    # replace punctuation by space
    txt = txt.translate({ord(c): " " for c in punctuation})

    # remove leading, trailing and repeated whitespace
    txt = txt.strip()
    txt = re.sub(r'\s+', ' ', txt)

    return txt