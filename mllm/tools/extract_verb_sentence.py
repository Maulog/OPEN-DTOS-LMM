import spacy
from spacy.matcher import Matcher

class VocabularyFilter:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")  # load English model
        self.matcher = Matcher(self.nlp.vocab)
        self.patterns = [                        # define patterns
            [{"POS": "VERB"}],
            [{"LEMMA" : {"IN" : ["up", "down", "left", "right", "front", "back"]}}],
            [{"LEMMA" : "side"}],
        ]
        
        self.matcher.add("my_pattern", self.patterns)

    def match_pattern(self, text):
        doc = self.nlp(text)
        matches = self.matcher(doc)
        if matches:
            return True
        return False

    def extract_actions(self, text_list):
        verblist = []
        for text in text_list:
            doc = self.nlp(text)
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]  # The matched span
                verblist.append(span.lemma_)

        return verblist

# vocabfilter = VocabularyFilter()

if __name__ == "__main__":
    texts = [
        "A red apple",
        "The car turned right at the traffic light.",
        "The cat sat on the mat.",
        "She painted a beautiful landscape.",
        "He was thinking about the future.",
        "The sun sets in the west.",
        "They were discussing the new project.",
        "A rainbow appeared after the rain.",
        "Alice runs swiftly through the park.",
        "Bob dances gracefully to the music.",
        "He kept the key just to the right of the door frame.",
        "Standing at the front of the classroom, the teacher began the lesson.",
        "She looked over her right shoulder to check for traffic.",
        "The stage was set with the band on the left and the speakers on the right.",
        "The hiker faced front, admiring the breathtaking mountain view.",
    ]

    vocabfilter = VocabularyFilter()
    action_texts = vocabfilter.extract_actions(texts)
    print(action_texts)

    # if vocabfilter.match_pattern(texts[1]):
    #     print("Matched")