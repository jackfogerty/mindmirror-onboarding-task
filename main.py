"""
STUDENT NAME: Jack Fogerty
STUDENT ID: 21080659
"""

from transformers import pipeline

# Load your chosen models here
def load_emotion_model():
    # i chose this emotion detection pipeline because it seemed accurate, and showed 7 different emotions, as well as showing values for each estimate
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    return classifier

def load_summarization_model():
    # i chose this summarization model because it seemed to be customizable enough for a user to use, with max length being changeable.
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    return summarizer

# Process journal entries and return emotion predictions
def detect_emotions(text_entries, emotion_model):
    # Implement logic to process entries and extract emotions
    emotions_list = []
    for text_entry in text_entries:
        emotions_list.append(find_emotion(emotion_model, text_entry))
    return emotions_list

def find_emotion(emotion_model, text_entry):
    #helper function for getting the highest score from the emotions list
    emotion_estimates = emotion_model(text_entry)
    most_emotion = max(emotion_estimates[0], key=lambda x: x['score'])['label']
    return most_emotion
    

# Generate summaries for journal entries
def summarize_entries(entries, summarizer):
    # Implement logic to summarize each entry
    summarized_entries = []
    for entry in entries:
        summarized_entry = summarizer(entry, max_length = 30, min_length = 5, do_sample = False)
        summarized_entries.append(summarized_entry)
    return summarized_entries

if __name__ == '__main__':
    # Load models
    emotion_model = load_emotion_model()
    summarizer = load_summarization_model()

    # Example input
    journal_entries = [
        "Felt anxious about my exam, but happy after completing it.",
        "It rained all day and I stayed inside feeling calm.",
        "Today was my first day of university, and I’m equal parts excited and overwhelmed. The campus is huge, and I got lost twice before finally finding my lecture hall. My professor seems nice, but the syllabus looks intense. Met a few people, but everything still feels so unfamiliar. Hoping tomorrow feels a little easier.",
        "It’s 1 AM, and I’m drowning in notes. I told myself I’d be better at time management this semester, yet here I am, cramming for a quiz. At least the library is quiet this late. I just hope I don’t sleep through my alarm tomorrow.",
        "I think I made a new friend today! We got paired for a group project, and we ended up grabbing coffee after class. It’s nice to finally have someone to talk to instead of just rushing between lectures. Maybe this place is starting to feel a little more like home.",
        "It hit me hard today—I really miss home. The food, my bed, even just sitting with my family doing nothing. I called my mom, and she told me to hang in there. She reminded me why I’m here, why I wanted this. I just need to push through.",
        "I got my essay back today, and I did way better than I expected! It feels good to know that all the late nights and effort actually paid off. I should celebrate, but honestly, I just want to take a nap.",
        "I dropped my entire tray in the middle of the cafeteria today. Just full-on, loud crash, food everywhere, all eyes on me. I wanted to disappear, but someone helped me clean up, and we ended up laughing about it. Maybe embarrassment isn’t the worst thing in the world.",
        "The stress is REAL. Midterms are coming up, and my brain feels like it’s melting. The library is packed, everyone is running on caffeine, and I have so much to study that I don’t even know where to start. One exam at a time, I guess.",
        "I took a walk around campus today instead of heading straight to my dorm. The sunset was beautiful, and for the first time in a while, I felt calm. No stress, no deadlines—just me, the cool breeze, and a quiet moment to breathe."
    ]

    # Apply pipelines
    emotions = detect_emotions(journal_entries, emotion_model)
    summaries = summarize_entries(journal_entries, summarizer)

    # Output results
    print("Emotion Predictions:", emotions)
    print("Summaries:", summaries)