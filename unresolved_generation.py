import random
from pathlib import Path

ROOT = Path(__file__)

def is_txt(path) -> bool:
    return str(path).split(".")[-1] == "txt"

def get_datafiles() -> list:
    """
    Returns list of stories(.txt file) in raw_story_file folder
    """    
    return [x for x in Path(ROOT.parent / "raw_story_file").iterdir() if is_txt(x)]

def genUnresolvedStoryLines(dataset: list) -> dict:
    """
    Removes random n lines from the end of a story to create unresolved storyliens
    """ 
    
    res = dict()
    for story_no, story in enumerate(dataset):     
        print(story_no, story)   
        temp = list()
        with open(story, "r", encoding="utf8") as data:               
            #Preprocessing - remove new line character and empty lines
            sentences = chr(32).join([paragraph for paragraph in data]).split(".")
            sentences = [x.replace("\n", "") for x in sentences if x != ""]        
            no_sentences = len(sentences)
            
            #Given number of lines will be random #See below 0 to 10% of Number of Sentences
            n = random.randint(0, int(0.1 * no_sentences))             

            #Create n text with n lines from the last removed            
            for i in range(1, n+1):
                data = sentences[:no_sentences-i]
                temp.append(data)

            #If the random number is 0, get the perfect data
            if len(res) == 0: temp = sentences

            res[story_no] = temp

    return res
  
if __name__ == "__main__":
    dataset = get_datafiles()
    res = genUnresolvedStoryLines(dataset=dataset)
    print(res.items())