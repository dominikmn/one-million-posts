import parsenvy


try:
    subscription_key = open(".azure_key").read().strip()
except:
    subscription_key = parsenvy.str("AZURE_KEY")

endpoint = "https://api.cognitive.microsofttranslator.com"

location = "westeurope"

if __name__=='__main__':
    print(endpoint)