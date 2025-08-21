import psycopg2
from langchain_ollama.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
# Load connection string for Neon server
conn_str = os.getenv('NEON_CONN_STR')

# Embedding model
ollama_embeddings_1024 = OllamaEmbeddings(model="embedding-cpu")

# Generated Q&A about an imaginary planet
planet_qa = {
    "What is the name of the planet?": "The planet is called Zephyra.",
    "Where is Zephyra located?": "It orbits a binary star system in the Andorian sector.",
    "How big is Zephyra compared to Earth?": "Zephyra is about 1.3 times larger than Earth.",
    "What is the atmosphere like?": "The atmosphere is rich in oxygen and neon, giving the sky a violet hue.",
    "Does Zephyra have moons?": "Yes, it has two moons named Luma and Kael.",
    "What is the average temperature?": "The average surface temperature is around 18°C, making it habitable.",
    "Are there oceans on Zephyra?": "Yes, vast turquoise oceans cover nearly 60% of its surface.",
    "What kind of gravity does it have?": "Zephyra’s gravity is slightly stronger than Earth’s, about 1.1 g.",
    "Does it have seasons?": "Yes, but the seasons are longer because of its wider orbit around the stars.",
    "What is unique about its sky?": "At night, both moons and the twin stars often appear together, creating breathtaking views.",
    "What creatures live there?": "Bioluminescent creatures called Aelari roam the forests and seas.",
    "What plants grow on Zephyra?": "Towering crystal-leafed trees that absorb starlight dominate its landscapes.",
    "Is there intelligent life?": "Yes, the Zephyrians, a peaceful race known for their deep connection to nature.",
    "What is Zephyra’s most famous landmark?": "The Shattered Peaks, floating mountains suspended by magnetic fields.",
    "How long is a day on Zephyra?": "A full day lasts about 30 Earth hours.",
    "How long is a year on Zephyra?": "A year is roughly 420 Earth days.",
    "Does Zephyra have storms?": "Yes, but they are gentle electrical storms that illuminate the skies with harmless lightning.",
    "What resources are found there?": "Zephyra is rich in a rare mineral called astralite, used for advanced energy systems.",
    "Is the planet colonized by humans?": "No, humans have only visited, but Zephyrians remain the primary inhabitants.",
    "What is the cultural belief of the Zephyrians?": "They believe every living thing shares a single ‘breath of starlight,’ uniting all life."
}

# Establish a connection, create a table, then insert data into it
with psycopg2.connect(conn_str) as conn:
    with conn.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS qa (
            id SERIAL PRIMARY KEY,
            q TEXT NOT NULL,
            a TEXT NOT NULL,
            q_embeddings VECTOR(1024) NOT NULL
        );""")

        for q, a in planet_qa.items():
            cur.execute("INSERT INTO qa (q, a, q_embeddings) VALUES (%s, %s, %s::vector)",
                        (q, a, ollama_embeddings_1024.embed_query(q)))

        conn.commit()