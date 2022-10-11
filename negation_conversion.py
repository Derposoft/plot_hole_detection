import json
import spacy

# The below function was taken from https://spacy.io/usage/linguistic-features#section-sbd
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text in [","]:
            doc[token.i+1].is_sent_start = True
    return doc

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(set_custom_boundaries, before="parser")
example_fn = "./data/test1_t1.json"

def load_examples(filename):
    with open(filename, 'r') as fp:
        example_list = json.load(fp)
    return example_list

def convert_negated_words(sentence):
    final_sentence = ""
    doc = nlp(sentence)
    my_sentences = [sent.text for sent in doc.sents]
    for str in my_sentences:
        neg_str = applyNegation(str)
        final_sentence += neg_str

    converted_sentence = final_sentence
    return converted_sentence.strip()

def applyNegation(sentence):
    doc = nlp(sentence)
    new_sentence = ""
    negation_words_list = ["no", "not", "never", "n't", "hardly", "barely", "none", "without", "nothing"]
    done = False
    for i, word in enumerate(doc):
        if word.lower_ in negation_words_list:
            new_sentence += word.text + " "
            if word.pos_ == "PART":
                part_negate = True
                keep_going = False
                if i + 1 >= len(doc):
                    done = True
                    break
                my_word = doc[i+1]
                if my_word.pos_ == "DET" or my_word.pos_ == "ADP" or my_word.pos_ == "NOUN":
                    for x in range(i + 1, len(doc)):
                        next_word = doc[x]
                        if (next_word.pos_ == "DET" or next_word.pos_ == "NOUN" or next_word.pos_ == "ADP"
                            or nounAncestryCheck(next_word)) and (part_negate == True and (next_word.lower_ not in negation_words_list)
                            and next_word.pos_ != "PUNCT"
                                 and next_word.text.isspace() == False):
                                new_sentence += "NOT_" + next_word.text + " "
                        else:
                            new_sentence += next_word.text + " "
                            part_negate = False
                    done = True
                if my_word.pos_ == "PRON":
                    head = my_word.head
                    head_lefts = [t for t in head.lefts]
                    head_rights = [t for t in head.rights]

                    if my_word  in head_lefts:
                        new_sentence += my_word.text + " "
                    elif my_word  in head_rights:
                        new_sentence += "NOT_" + my_word.text + " "

                    if i + 2 >= len(doc):
                        done = True
                        break
                    for x in range(i + 2, len(doc)):
                        next_word = doc[x]
                        if (next_word.pos_ == "DET" or next_word.pos_ == "NOUN" or next_word.pos_ == "ADP"
                            or nounAncestryCheck(next_word)) and (part_negate == True and (next_word.lower_ not in negation_words_list)
                            and next_word.pos_ != "PUNCT"
                                 and next_word.text.isspace() == False):
                                new_sentence += "NOT_" + next_word.text + " "
                        elif next_word.pos_ == "AUX":
                            new_sentence += next_word.text + " "
                            for y in range(x + 1, len(doc)):
                                next_word = doc[y]
                                if (next_word.pos_ == "ADJ" or next_word.pos_ == "NOUN" or next_word.pos_ == "PRON" or nounAncestryCheck(next_word)) \
                                    and (next_word.lower_ not in negation_words_list) and next_word.pos_ != "PUNCT" and part_negate \
                                    and next_word.text.isspace() == False:
                                    new_sentence += "NOT_" + next_word.text + " "
                                else:
                                    new_sentence += next_word.text + " "
                                    part_negate = False
                            done = True
                            break
                        else:
                            new_sentence += next_word.text + " "
                            part_negate = False
                    done = True
                if my_word.pos_ == "ADJ" or my_word.pos_ == "ADV":
                    for x in range(i + 1, len(doc)):
                        next_word = doc[x]
                        if (next_word.pos_ == "ADJ" or next_word.pos_ == "ADV" or next_word.pos_ == "NOUN" or nounAncestryCheck(next_word)) \
                                and (next_word.lower_ not in negation_words_list) and next_word.pos_ != "PUNCT" and part_negate \
                                and next_word.text.isspace() == False:
                            new_sentence += "NOT_" + next_word.text + " "
                        else:
                            new_sentence += next_word.text + " "
                            part_negate = False
                    done = True
                if my_word.pos_ == "CCONJ":
                    for x in range(i + 1, len(doc)):
                        next_word = doc[x]
                        new_sentence += next_word.text + " "
                        #part_negate = False
                    done = True
                if my_word.pos_ == "AUX":
                    new_sentence += my_word.text + " "
                    if i+2 >= len(doc):
                        done = True
                        break
                    for x in range(i+2, len(doc)):
                        next_word = doc[x]
                        if (next_word.pos_ == "ADV" or next_word.pos_ == "ADJ" or next_word.pos_ == "NOUN" or next_word.pos_ == "PRON" or nounAncestryCheck(next_word)) \
                                and (next_word.lower_ not in negation_words_list) and next_word.pos_ != "PUNCT" and part_negate \
                                and next_word.text.isspace() == False:
                            new_sentence += "NOT_" + next_word.text + " "
                        else:
                            new_sentence += next_word.text + " "
                            part_negate = False
                    done = True
                elif my_word.pos_ == "VERB":
                    for x in range(i + 1, len(doc)):
                        next_word = doc[x]
                        if (next_word.pos_ == "VERB" and part_negate) or (
                                keep_going and (next_word.pos_ != "PUNCT") and (
                                next_word.lower_ not in negation_words_list)) and next_word.text.isspace() == False:
                            new_sentence += "NOT_" + next_word.text + " "
                            keep_going = True
                        else:
                            new_sentence += next_word.text + " "
                            part_negate == False
                    done = True

            elif word.pos_ == "ADV":
                part_negate = True
                if i + 1 >= len(doc):
                    done = True
                    break
                my_word = doc[i + 1]
                if my_word.pos_ == "AUX":
                    new_sentence += my_word.text + " "
                    if i+2 >= len(doc):
                        done = True
                        break
                    for x in range(i+2, len(doc)):
                        next_word = doc[x]
                        if (next_word.pos_ == "NOUN" or next_word.pos_ == "PRON" or nounAncestryCheck(next_word)) \
                                and (next_word.lower_ not in negation_words_list) and next_word.pos_ != "PUNCT" and part_negate \
                                and next_word.text.isspace() == False:
                            new_sentence += "NOT_" + next_word.text + " "
                        elif (next_word.pos_ == "VERB" and part_negate) or (
                                 (next_word.pos_ != "PUNCT") and (
                                next_word.lower_ not in negation_words_list)) and next_word.text.isspace() == False:
                            new_sentence += "NOT_" + next_word.text + " "
                        else:
                            new_sentence += next_word.text + " "
                            part_negate = False
                    done = True
                elif my_word.pos_ == "ADJ":
                    for x in range(i + 1, len(doc)):
                        next_word = doc[x]
                        if (next_word.pos_ == "ADJ" or next_word.pos_ == "PART" or next_word.pos_ == "VERB" or next_word.pos_ == "NOUN" or nounAncestryCheck(next_word)) \
                                and (next_word.lower_ not in negation_words_list) and next_word.pos_ != "PUNCT" and part_negate \
                                and next_word.text.isspace() == False:
                            new_sentence += "NOT_" + next_word.text + " "
                        else:
                            new_sentence += next_word.text + " "
                            part_negate = False
                    done = True
                elif my_word.pos_ == "VERB":
                    for x in range(i + 1, len(doc)):
                        next_word = doc[x]
                        if (next_word.pos_ == "VERB" and part_negate) or (
                                keep_going and (next_word.pos_ != "PUNCT") and (
                                next_word.lower_ not in negation_words_list)) and next_word.text.isspace() == False:
                            new_sentence += "NOT_" + next_word.text + " "
                            keep_going = True
                        else:
                            new_sentence += next_word.text + " "
                            part_negate == False
                    done = True
                if my_word.pos_ == "DET":
                    for x in range(i + 1, len(doc)):
                        next_word = doc[x]
                        if (next_word.pos_ == "ADJ" or next_word.pos_ == "ADV" or next_word.pos_ == "NOUN" or nounAncestryCheck(next_word)) \
                                and (next_word.lower_ not in negation_words_list) and next_word.pos_ != "PUNCT" and part_negate \
                                and next_word.text.isspace() == False:
                            new_sentence += "NOT_" + next_word.text + " "
                        else:
                            new_sentence += next_word.text + " "
                            part_negate = False
                    done = True
            elif word.pos_ == "DET":
                part_negate = True
                keep_going = False
                if i + 1 >= len(doc):
                    done = True
                    break
                my_word = doc[i+1]
                if my_word.pos_ == "DET" or my_word.pos_ == "ADP" or my_word.pos_ == "NOUN":
                    for x in range(i + 1, len(doc)):
                        next_word = doc[x]
                        if (next_word.pos_ == "DET" or next_word.pos_ == "NOUN" or next_word.pos_ == "ADP" or next_word.pos_ == "VERB") \
                                and (part_negate == True and (
                                next_word.lower_ not in negation_words_list) and next_word.pos_ != "PUNCT"
                                     and next_word.text.isspace() == False):
                            new_sentence += "NOT_" + next_word.text + " "
                        elif (next_word.pos_ == "CCONJ") \
                                and (part_negate == True and (
                                next_word.lower_ not in negation_words_list) and next_word.pos_ != "PUNCT"
                                     and next_word.text.isspace() == False):
                            if x + 1 >= len(doc):
                                break
                                done = True
                            if next_word.head == doc[x+1].head:
                                new_sentence += "NOT_" + next_word.text + " "
                            else:
                                new_sentence += next_word.text + " "
                                part_negate = False
                        else:
                            new_sentence += next_word.text + " "
                            part_negate = False
                    done = True
                elif my_word.pos_ != "PUNCT":
                    new_sentence += "NOT_" + my_word.text + " "
                else:
                        new_sentence += my_word.text + " "
                done = True
            elif word.pos_ == "ADP":
                part_negate = True
                keep_going = False
                if i + 1 >= len(doc):
                    done = True
                    break
                my_word = doc[i+1]
                if my_word.pos_ == "DET" or my_word.pos_ == "ADP" or my_word.pos_ == "NOUN":
                    for x in range(i + 1, len(doc)):
                        next_word = doc[x]
                        if (next_word.pos_ == "DET" or next_word.pos_ == "NOUN" or next_word.pos_ == "ADP" or next_word.pos_ == "VERB") \
                                and (part_negate == True and (
                                next_word.lower_ not in negation_words_list) and next_word.pos_ != "PUNCT"
                                     and next_word.text.isspace() == False):
                            new_sentence += "NOT_" + next_word.text + " "
                        elif (next_word.pos_ == "CCONJ") \
                                and (part_negate == True and (
                                next_word.lower_ not in negation_words_list) and next_word.pos_ != "PUNCT"
                                     and next_word.text.isspace() == False):
                            if x + 1 >= len(doc):
                                break
                                done = True
                            if next_word.head == doc[x+1].head:
                                new_sentence += "NOT_" + next_word.text + " "
                            else:
                                new_sentence += next_word.text + " "
                                part_negate = False
                        else:
                            new_sentence += next_word.text + " "
                            part_negate = False
                    done = True
                elif my_word.pos_ != "PUNCT":
                    new_sentence += "NOT_" + my_word.text + " "
                else:
                        new_sentence += my_word.text + " "
                done = True
            elif word.pos_ == "NOUN":
                part_negate = True
                keep_going = False
                if i + 1 >= len(doc):
                    done = True
                    break
                my_word = doc[i+1]
                if my_word.pos_ == "DET" or my_word.pos_ == "ADP" or my_word.pos_ == "NOUN" or my_word.pos_ == "AUX":
                    for x in range(i + 1, len(doc)):
                        next_word = doc[x]
                        if (next_word.pos_ == "DET" or next_word.pos_ == "NOUN" or next_word.pos_ == "ADP" or next_word.pos_ == "ADV" or next_word.pos_ == "VERB"
                            or next_word.pos_ == "PROPN" or next_word.pos_ == "AUX" or next_word.pos_ == "VERB") and (part_negate == True and ( next_word.lower_ not in negation_words_list)
                            and next_word.pos_ != "PUNCT" and next_word.text.isspace() == False):
                            new_sentence += "NOT_" + next_word.text + " "
                        elif (next_word.pos_ == "CCONJ") \
                                and (part_negate == True and (
                                next_word.lower_ not in negation_words_list) and next_word.pos_ != "PUNCT"
                                     and next_word.text.isspace() == False):
                            if x + 1 >= len(doc):
                                break
                                done = True
                            if next_word.head == doc[x+1].head:
                                new_sentence += "NOT_" + next_word.text + " "
                            else:
                                new_sentence += next_word.text + " "
                                part_negate = False
                        else:
                            new_sentence += next_word.text + " "
                            part_negate = False
                    done = True
                else:
                        new_sentence += my_word.text + " "
                done = True
            elif word.pos_ == "PRON":
                part_negate = True
                keep_going = False
                if i + 1 >= len(doc):
                    done = True
                    break
                my_word = doc[i+1]
                if my_word.pos_ == "ADJ" or my_word.pos_ == "DET" or my_word.pos_ == "ADP" or my_word.pos_ == "NOUN":
                    for x in range(i + 1, len(doc)):
                        next_word = doc[x]
                        if (next_word.pos_ == "PRON" or next_word.pos_ == "ADP" or next_word.pos_ == "AUX" or next_word.pos_ == "VERB"
                            or next_word.pos_ == "PRON" or next_word.pos_ == "ADJ") and (part_negate == True and ( next_word.lower_ not in negation_words_list)
                            and next_word.pos_ != "PUNCT" and next_word.text.isspace() == False):
                            new_sentence += "NOT_" + next_word.text + " "
                        elif (next_word.pos_ == "CCONJ") \
                                and (part_negate == True and (
                                next_word.lower_ not in negation_words_list) and next_word.pos_ != "PUNCT"
                                     and next_word.text.isspace() == False):
                            if x + 1 >= len(doc):
                                break
                                done = True
                            if next_word.head == doc[x+1].head:
                                new_sentence += "NOT_" + next_word.text + " "
                            else:
                                new_sentence += next_word.text + " "
                                part_negate = False
                        else:
                            new_sentence += next_word.text + " "
                            part_negate = False
                    done = True
                else:
                        new_sentence += my_word.text + " "
                done = True
        if done == True:
            break
        else:
            new_sentence += word.text + " "
    return new_sentence

def nounAncestryCheck(word):
    if word.head.pos_ == "NOUN":
        return True
    else:
        if word.head.text != word.text:
            return nounAncestryCheck(word.head)
    return False

def test_examples(filename):
    examples = load_examples(filename)
    total = len(examples)
    correct = 0
    for example in examples:
        print("\nPhrase:", example['S'])
        converted = convert_negated_words(example['S'])
        print("Conversion:", converted)
        print("Solution:  ", example['N'])

        if converted.strip() == example['N'].strip():
            correct += 1
    print("\nAccuracy: ", (correct * 100) / total, "%")

if __name__ == '__main__':
    test_examples(example_fn)
