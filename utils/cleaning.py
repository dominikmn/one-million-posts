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

def normalize(txt:str, url_emoji_dummy:bool=False, pure_words:bool=True):
    """Normalizes a string
    
    1. Applies .lower()
    2. Replaces URLs by space (or 'URL')
    3. Replaces emojis by space (or 'EMOJI')
    4. Replaces any punctuation by space (or removes repeated punctuation)
    5. Removes leading, trailing, repeated spaces

    Args:
        txt: str 
        url_emoji_dummy: bool
            If True urls and emojist will be replaced by 'URL' and 'EMOJI' respecitively.
            If False, they will be replaced by a space character. 
        pure_words: bool
            If True, str.lower() and .translate(...) are applied on txt to make the string lower case and remove any punctuation respectively. 
            If False, .lower() is skipped. Puncuation is still removed but only repeated occurences.

    Returns:
        txt: str object in normalized format.
    """
    if pure_words:
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
    s += r"(?:[<(]?[oO]_[oO][)>]?)"    # 'eastern' emoji.
    s += r")"                    # end emoticon
    s += r"(?=\s|$)"             # white space or end required after
    emoticon_re = re.compile(s)
    txt = emoticon_re.sub(emoji_dummy, txt)  #replace with 'EMOTICON but keep preceeding and trailing space/linefeed

    if pure_words:
        # replace punctuation by space
        txt = txt.translate({ord(c): " " for c in punctuation})
    else:
        # remove repeated punctuation
        last = None
        output = []
        for c in txt:
            if c != last:
                if c in punctuation:
                    last = c
                else:
                    last = None
                output.append(c)
        txt = ''.join(output)

    # remove leading, trailing and repeated space
    txt = txt.strip()
    txt = re.sub(r'\s+', ' ', txt)

    return txt