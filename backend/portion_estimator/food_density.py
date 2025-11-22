# portion_estimator/food_density.py

"""
Food density database for portion estimation
Density in g/cm³
"""

FOOD_DENSITY = {
    # Fruits
    "apple": 0.65,
    "banana": 0.94,
    "orange": 0.70,
    "grapes": 0.90,
    
    # Meats & Proteins
    "chicken breast": 1.05,
    "steak": 1.10,
    "beef carpaccio": 1.05,
    "beef tartare": 1.00,
    "baby back ribs": 0.95,
    "grilled salmon": 1.05,
    "fried calamari": 0.70,
    "crab cakes": 0.85,
    
    # Fast Food
    "burger": 0.90,
    "hamburger": 0.90,
    "hot dog": 0.85,
    "pizza": 0.55,
    "french fries": 0.40,
    "chicken wings": 0.80,
    "chicken quesadilla": 0.70,
    
    # Asian Cuisine
    "sushi": 0.75,
    "ramen": 0.95,
    "bibimbap": 0.85,
    "fried rice": 0.85,
    "dumplings": 0.90,
    "gyoza": 0.90,
    "edamame": 0.75,
    
    # Salads
    "salad": 0.20,
    "caesar salad": 0.25,
    "greek salad": 0.30,
    "caprese salad": 0.35,
    "beet salad": 0.30,
    
    # Pasta & Noodles
    "pasta": 0.60,
    "noodles": 0.80,
    "gnocchi": 0.75,
    
    # Breakfast
    "eggs benedict": 0.90,
    "breakfast burrito": 0.85,
    "french toast": 0.50,
    "pancakes": 0.45,
    
    # Sandwiches
    "sandwich": 0.45,
    "club sandwich": 0.50,
    "grilled cheese sandwich": 0.60,
    "croque madame": 0.65,
    
    # Desserts
    "ice cream": 0.55,
    "cheesecake": 0.80,
    "chocolate cake": 0.65,
    "carrot cake": 0.60,
    "cup cakes": 0.50,
    "donuts": 0.40,
    "chocolate mousse": 0.70,
    "creme brulee": 0.90,
    "tiramisu": 0.75,
    "bread pudding": 0.70,
    "cannoli": 0.65,
    "baklava": 0.75,
    "beignets": 0.45,
    "churros": 0.40,
    
    # Indian
    "samosa": 0.70,
    "chicken curry": 0.90,
    
    # Appetizers & Sides
    "garlic bread": 0.35,
    "bruschetta": 0.40,
    "deviled eggs": 0.95,
    "hummus": 1.00,
    "guacamole": 0.90,
    "falafel": 0.70,
    "cheese plate": 1.05,
    
    # Soups
    "clam chowder": 0.95,
    "french onion soup": 0.90,
    
    # Pies
    "apple pie": 0.65,
    
    # Other
    "frozen yogurt": 0.60,
    "ceviche": 0.85,
}

# Default density for unknown foods
DEFAULT_DENSITY = 0.80

print(f"✅ Loaded {len(FOOD_DENSITY)} food densities")
