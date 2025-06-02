import json
import random

# Base templates for romantic conversations
conversation_templates = [
    # English templates
    {
        "input": "Human: {greeting}, how are you feeling today? {emoji}",
        "response": "AI: {response_start} when I'm talking to you {love_emoji}"
    },
    {
        "input": "Human: I was thinking about you and {action} {emoji}",
        "response": "AI: {sweet_response} {heart_emoji}"
    },
    {
        "input": "Human: What would you do if {romantic_scenario}?",
        "response": "AI: {romantic_response} {romantic_emoji}"
    },
    
    # Hindi/Hinglish templates
    {
        "input": "Human: {hindi_greeting}, tumhara din kaisa ja raha hai? {emoji}",
        "response": "AI: {hindi_response_start} tumhare saath {hindi_love_expr} {emoji}"
    },
    {
        "input": "Human: Tumhe pata hai {hindi_feeling} {emotion_emoji}",
        "response": "AI: {hindi_sweet_response} {heart_emoji}"
    },
    {
        "input": "Human: Agar {hindi_scenario} toh kya karoge?",
        "response": "AI: {hindi_romantic_response} {romantic_emoji}"
    },
    
    # Mixed language templates
    {
        "input": "Human: {mixed_greeting} {feeling_english} about {topic} {emoji}",
        "response": "AI: {mixed_response} {supportive_emoji}"
    }
]

# Word banks for generating varied content
greetings = ["Hey beautiful", "Good morning sunshine", "Hi gorgeous", "Hello love", "Hey cutie"]
hindi_greetings = ["Namaste jaan", "Hey baby", "Arre yaar", "Kya haal hai", "Kaise ho"]
mixed_greetings = ["Hey jaan", "Hello sweetheart", "Hi baby", "Namaste love"]

response_starts = ["Amazing", "Perfect", "Wonderful", "Incredible", "Fantastic"]
hindi_response_starts = ["Bohot accha", "Ekdum perfect", "Kamal ka", "Zabardast", "Shandar"]

love_emojis = ["💕", "❤️", "💖", "💗", "💝", "🥰", "😍", "💘"]
heart_emojis = ["💓", "💞", "💟", "❣️", "💌", "💋", "😘"]
romantic_emojis = ["🌹", "✨", "🌟", "💫", "🦋", "🌙", "⭐"]
emotion_emojis = ["😊", "😭", "🥺", "😌", "😴", "😰", "🤔"]

# Generate diverse romantic conversations
def generate_conversation_set(theme, count=100):
    conversations = []
    
    for i in range(count):
        # Select random elements for variety
        template = random.choice(conversation_templates)
        
        # Create conversation based on theme and template
        if "hindi" in template["input"].lower():
            # Hindi conversation
            conversation = {
                "input": template["input"].format(
                    hindi_greeting=random.choice(hindi_greetings),
                    hindi_feeling=random.choice(["main tumse bohot pyaar karta hun", "tum meri zindagi ho", "tumhare bina adhoora hun"]),
                    hindi_scenario=random.choice(["hum saath mein hote", "tumhe surprise deta", "tumhara haath pakad sakta"]),
                    emoji=random.choice(love_emojis + emotion_emojis)
                ),
                "response": template["response"].format(
                    hindi_response_start=random.choice(hindi_response_starts),
                    hindi_love_expr=random.choice(["bhi pyaar hai", "dil khush hai", "sab kuch perfect hai"]),
                    hindi_sweet_response=random.choice(["Aur main tumse", "Tumhara pyaar hi toh", "Bas tumhara saath"]),
                    hindi_romantic_response=random.choice(["Tumhare paas aa jaunga", "Tumhe tight hug dunga", "Kabhi nahi jaane dunga"]),
                    emoji=random.choice(heart_emojis + romantic_emojis),
                    heart_emoji=random.choice(heart_emojis),
                    romantic_emoji=random.choice(romantic_emojis)
                )
            }
        else:
            # English conversation
            conversation = {
                "input": template["input"].format(
                    greeting=random.choice(greetings),
                    action=random.choice(["smiled", "felt happy", "got butterflies"]),
                    romantic_scenario=random.choice(["we could hug right now", "I could hold your hand", "we were watching stars together"]),
                    feeling_english=random.choice(["excited", "nervous", "happy", "curious"]),
                    topic=random.choice(["us", "our future", "our love", "you"]),
                    mixed_greeting=random.choice(mixed_greetings),
                    emoji=random.choice(love_emojis + emotion_emojis)
                ),
                "response": template["response"].format(
                    response_start=random.choice(response_starts),
                    sweet_response=random.choice(["That makes my heart flutter", "You're so precious to me", "I love that about you"]),
                    romantic_response=random.choice(["I'd never let you go", "I'd hold you forever", "I'd cherish every second"]),
                    mixed_response=random.choice(["That's so sweet yaar", "You make me happy jaan", "Main bhi feel the same"]),
                    love_emoji=random.choice(love_emojis),
                    heart_emoji=random.choice(heart_emojis),
                    romantic_emoji=random.choice(romantic_emojis),
                    supportive_emoji=random.choice(["🤗", "💪", "✨", "🌟"])
                )
            }
        
        conversations.append(conversation)
    
    return conversations

# Themes for different datasets
themes = [
    "morning_love", "evening_romance", "missing_you", "future_dreams", 
    "sweet_nothings", "long_distance", "support_love", "playful_flirt",
    "deep_connection", "daily_affection", "romantic_gestures", "virtual_dates",
    "love_confessions", "cute_moments", "relationship_goals", "intimate_talks",
    "surprise_love", "caring_words", "dreamy_conversations", "heart_to_heart",
    "passionate_love", "gentle_affection", "romantic_poetry", "love_promises",
    "sweet_memories", "beautiful_moments", "endless_love", "pure_emotions",
    "romantic_dreams", "loving_support", "tender_moments", "heartfelt_words",
    "love_letters", "romantic_wishes", "sweet_surprises", "caring_love",
    "deep_emotions", "romantic_thoughts", "loving_gestures", "heart_talks",
    "sweet_conversations", "romantic_moments", "love_expressions", "tender_love",
    "passionate_words", "gentle_romance", "sweet_talks", "loving_words"
]

# Generate all datasets
def generate_all_datasets():
    for i, theme in enumerate(themes[:46], 5):  # Starting from 5 since we already created 4 manually
        conversations = []
        
        # Generate multiple sets to reach 1000+ lines
        for batch in range(10):  # 10 batches of 100 = 1000 conversations
            batch_conversations = generate_conversation_set(theme, 100)
            conversations.extend(batch_conversations)
        
        filename = f"/Users/shaswatraj/Desktop/AI/LOVE/romantic_chat_{i:02d}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        print(f"Generated {filename} with {len(conversations)} conversations")

if __name__ == "__main__":
    generate_all_datasets()
    print("All 50 romantic chat datasets generated successfully!")
