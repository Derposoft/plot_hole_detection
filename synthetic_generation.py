text = "Pygmalion, the mythical king of Cyprus, had many problems when dating women. He always seemed to accept dates from the wrong women. Some were rude, others were selfish; he was revolted by the faults nature had placed in these women. It left him feeling very depressed. He eventually came to despise the female gender so much that he decided he would never marry any maiden. For comfort and solace, he turned to the arts, finding his talent in sculpture. Using exquisite skills, he carved a statue out of ivory that was so resplendent and delicate no maiden could compare with its beauty. This statute was the perfect resemblance of a living maiden. Pygmalion fell in love with his creation and often laid his hand upon the ivory statute as if to reassure himself it was not living. He named the ivory maiden Galatea and adorned her lovely figure with women’s robes and placed rings on her fingers and jewels about her neck. At the festival of Aphrodite, which was celebrated with great relish throughout all of Cyprus, lonely Pygmalion lamented his situation. When the time came for him to play his part in the processional, Pygmalion stood by the altar and humbly prayed: “If you gods can give all things, may I have as my wife, I pray…” he did not dare say “the ivory maiden” but instead said: “one like the ivory maiden.” Aphrodite, who also attended the festival, heard his plea and she also knew of the thought he had wanted to utter. Showing her favor, she caused the altar’s flame to flare up three times, shooting a long flame of fire into the still air. After the day’s festivities, Pygmalion returned home and kissed Galatea as was his custom. At the warmth of her kiss, he started as if stung by a hornet. The arms that were ivory now felt soft to his touch and when he softly pressed her neck the veins throbbed with life. Humbly raising her eyes, the maiden saw Pygmalion and the light of day simultaneously. Aphrodite blessed the happiness and union of this couple with a child. Pygmalion and Galatea named the child Paphos, for which the city is known until this day. Story Location Clue: Pygmalion and Galatea lived out their days in the city of Paphos located west of the Troodos Mountain Range along the western coast of Cyprus. This city is also north and west of Aphrodite’s Rock. "

import numpy as np
from copy import deepcopy
#n number of synthetic datapoints of Continuity Errors --> negating a sentance prior
n = 10

from negation_conversion import applyNegation
def getContiuityErrors(document, n):
    sentences = document.split(".")
    #print(sentences)
    #print()
    samples = np.random.choice(range(len(sentences)), n, replace=False)
    #print(samples)
    X = []
    negate = {"was", "is", "are", "am"}
    for sample in samples:
        X.append(deepcopy(sentences))

        #X[-1][sample] = "".join([word if word not in negate or not word.endswith("ed") else word + " not " for word in X[-1][sample]])
        test = []
        for word in X[-1][sample].split(" "):
            #print(word)
            #print("word here: ", word)
            if word.endswith("ed"):
                #print("HERE")
                test.append("was not "+word)
            elif word in negate:
                test.append(word + " not")
            else:
                test.append(word)

        X[-1][sample] = " ".join(test)
    y = samples
    return X,y


X,y = getContiuityErrors(text, n)

for t in enumerate(text.split(".")):
    print(t)

print()

print(y[0])
print(X[0][y[0]])
print(text.split(".")[y[0]])