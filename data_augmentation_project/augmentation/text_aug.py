# augmentation/text_aug.py

import nlpaug.augmenter.word as naw

class TextAugmentor:
    def __init__(self):
        self.synonym_aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.3)

    def augment(self, text):
        return self.synonym_aug.augment(text)