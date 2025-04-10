from augmentation.image_aug import augment_image

augment_image("golden.jpg", "golden_aug.jpg")

# test.py

from augmentation import ImageAugmentor, TextAugmentor, AudioAugmentor

# 🖼️ Test Image
img_aug = ImageAugmentor()
img_aug.augment("sample.jpg", "aug_sample.jpg")

# 📝 Test Text
txt_aug = TextAugmentor()
original_text = "The quick brown fox jumps over the lazy dog"
augmented_text = txt_aug.augment(original_text)
print("Original:", original_text)
print("Augmented:", augmented_text)

# 🔊 Test Audio
aud_aug = AudioAugmentor()
aud_aug.augment("sample.wav", "aug_sample.wav")
