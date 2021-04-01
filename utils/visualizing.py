from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_wordcloud_freq(series, colormap='BuGn'):
    '''Print a wordcloud from a series or dictionary with word frequencies
    Arguments: series - a series or dict containing word frequencies, colormap = a matplotlib colormap
    '''
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    min_font_size = 10).generate_from_frequencies(series) 
    wordcloud.recolor(colormap=colormap)    
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    
def plot_wordcloud_string(series, colormap='BuGn'):
    '''Print a wordcloud from a string
    Arguments: series - a string, colormap = a matplotlib colormap
    '''
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    min_font_size = 10).generate_from_frequencies(string) 
    wordcloud.recolor(colormap=colormap)    
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 