import json
import numpy as np
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')

def accuracy_exact_match(pred, target):
    '''
    Checks if the target is a substring of the prediction. Makes pred and target strings lowercase first. 
    '''
    comp = np.array([t.lower() in p.lower() for t, p in zip(target, pred)])
    # comp = [t == p for t, p in zip(target, pred)]
    return comp

def accuracy_synonmy(pred, target):
    '''
    Checks if the target or is a substring of the prediction or a synanym of the prediction. Makes pred adn target strings lowercase first.
    '''
    comp = [any(syn.lower() in p.lower() for syn in get_synonyms(t.lower())) for p, t in zip(pred, target)]
    # comp = np.array([p.lower() in get_synonyms(t) for p, t in zip(pred, target)])
    return comp
    # return comp.sum() / len(comp)

def get_synonyms(word):
    synonyms = [word]
    for syn in wordnet.synsets(word.lower()):
        for lm in syn.lemmas():
            synonyms.append(lm.name().replace('_', ' '))
    # synonyms = ' '.join(synonyms)
    return synonyms


if __name__=="__main__":
    f = json.load(open('pred_flam_vqa.json'))
    pred = []
    target = []
    for i in range(2,len(f)):
        pred.extend(f[i]['pred_answers'])
        target.extend(f[i]['answers'])
    exact_match = accuracy_exact_match(pred, target)
    syn = accuracy_synonmy(pred, target)
    print("Exact Match: ", sum(exact_match)/len(pred))
    print("Almost Correctness: ", sum(syn)/len(pred))