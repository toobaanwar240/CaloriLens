# classification/class_labels.py

# class_labels.py

# class_labels.py

# Model 1: 11 classes (your custom trained model)
MODEL1_LABELS = [
    "beef_carpaccio",
    "bibimbap",
    "caesar_salad",
    "hamburger",
    "hot_dog",
    "ice_cream",
    "pizza",           # ⭐ Unique to Model 1
    "ramen",           # ⭐ Unique to Model 1
    "samosa",          # ⭐ Unique to Model 1
    "steak",           # ⭐ Unique to Model 1
    "sushi"            # ⭐ Unique to Model 1
]

# Model 2: 49 classes (Food-101)
MODEL2_LABELS = [
    "apple_pie",
    "baby_back_ribs",
    "baklava",
    "beef_carpaccio",      # Overlap
    "beef_tartare",
    "beet_salad",
    "beignets",
    "bibimbap",            # Overlap
    "bread_pudding",
    "breakfast_burrito",
    "bruschetta",
    "caesar_salad",        # Overlap
    "cannoli",
    "caprese_salad",
    "carrot_cake",
    "ceviche",
    "cheese_plate",
    "cheesecake",
    "chicken_curry",
    "chicken_quesadilla",
    "chicken_wings",
    "chocolate_cake",
    "chocolate_mousse",
    "clam_chowder",
    "club_sandwich",
    "crab_cakes",
    "creme_brulee",
    "croque_madame",
    "cup_cakes",
    "deviled_eggs",
    "donuts",
    "dumplings",
    "edamame",
    "eggs_benedict",
    "falafel",
    "fried_calamari",
    "fried_rice",
    "frozen_yogurt",
    "garlic_bread",
    "gnocchi",
    "greek_salad",
    "grilled_cheese_sandwich",
    "grilled_salmon",
    "guacamole",
    "gyoza",
    "hamburger",           # Overlap
    "hot_dog",             # Overlap
    "hummus",
    "ice_cream"            # Overlap
]

# Unique classes in each model
UNIQUE_TO_MODEL1 = ["pizza", "ramen", "samosa", "steak", "sushi"]
UNIQUE_TO_MODEL2 = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_tartare", 
    "beet_salad", "beignets", "bread_pudding", "breakfast_burrito",
    "bruschetta", "cannoli", "caprese_salad", "carrot_cake", "ceviche",
    "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame",
    "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame",
    "eggs_benedict", "falafel", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad",
    "grilled_cheese_sandwich", "grilled_salmon", "guacamole",
    "gyoza", "hummus"
]

# Overlapping classes (present in both)
OVERLAPPING_LABELS = [
    "beef_carpaccio", "bibimbap", "caesar_salad", 
    "hamburger", "hot_dog", "ice_cream"
]

# All unique classes combined (54 total)
ALL_LABELS = sorted(list(set(MODEL1_LABELS + MODEL2_LABELS)))

NUM_CLASSES_MODEL1 = len(MODEL1_LABELS)  # 11
NUM_CLASSES_MODEL2 = len(MODEL2_LABELS)  # 49
NUM_CLASSES_TOTAL = len(ALL_LABELS)      # 54

print(f"Model 1: {NUM_CLASSES_MODEL1} classes")
print(f"Model 2: {NUM_CLASSES_MODEL2} classes")
print(f"Total unique: {NUM_CLASSES_TOTAL} classes")
print(f"Unique to Model 1: {UNIQUE_TO_MODEL1}")
print(f"Overlapping: {OVERLAPPING_LABELS}")

