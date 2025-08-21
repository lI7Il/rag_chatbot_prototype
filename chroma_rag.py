from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_chroma import Chroma
from uuid import uuid4

# Initialize Ollama models
ollama_embeddings_1024 = OllamaEmbeddings(model="embedding-cpu")
llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0,
)

# Creating a presistant Chroma DB using LangChain
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=ollama_embeddings_1024,
    persist_directory="./.db",
    collection_configuration={"hnsw": {"space": "cosine"}} # Using cosine similarity
)

# Resets the DB (since it's presistant it hold data from last run) !FOR EXPERMENTING
vector_store.reset_collection()

# List of facts
facts = [
    "Bananas are berries, but strawberries are not.",
    "Sharks existed before trees.",
    "Octopuses have three hearts.",
    "Sloths can hold their breath longer than dolphins.",
    "The Eiffel Tower can grow over 6 inches taller in summer due to heat expansion.",
    "A day on Venus is longer than a year on Venus.",
    "Honey never spoils — archaeologists have eaten 3,000-year-old honey.",
    "Wombat poop is cube-shaped.",
    "Your stomach gets a new lining every 3 to 4 days.",
    "The inventor of the Pringles can is buried in one.",
    "Some turtles can breathe through their butts.",
    "Shakespeare invented more than 1,700 words.",
    "A group of flamingos is called a flamboyance.",
    "The moon has moonquakes.",
    "There are more stars in the universe than grains of sand on Earth."
]
facts_about_character = [
    "Liora Thalewind, a skyship navigator, can fall asleep instantly when hearing thunder.",
    "She has a scar shaped like a crescent moon on her left palm, but she doesn’t remember how she got it.",
    "Liora secretly collects buttons from every city she visits, storing them in a tin box.",
    "She is afraid of mirrors, convinced they sometimes show her a future version of herself.",
    "Her favorite food is roasted chestnuts, which she eats even in the middle of summer.",
    "Liora once tamed a wild stormhawk by singing to it during a lightning storm.",
    "She cannot whistle, no matter how hard she tries.",
    "Every year on her birthday, she receives an anonymous letter written in silver ink.",
    "She carries a compass that always points to the person she trusts most, not north.",
    "Despite being a navigator, she gets lost easily on land when there are no stars to guide her."
]
creature_facts = [
    "The Lumivora is a nocturnal creature that glows faintly in the dark.",
    "It has four translucent wings that shimmer like glass when hit by moonlight.",
    "The creature communicates through low humming sounds that can calm nearby animals.",
    "Its diet consists mostly of glowing moss and dew drops found in deep forests.",
    "Legends say the Lumivora can guide lost travelers by creating a trail of light.",
    "It has a natural defense mechanism: when threatened, its body becomes nearly invisible.",
    "Despite its fragile appearance, it can fly for several days without resting.",
    "Young Lumivoras are born with no glow, and their light grows brighter as they age.",
    "They are rarely seen in groups, preferring a solitary life high in the treetops.",
    "In folklore, spotting a Lumivora is believed to bring clarity to difficult decisions."
]

# Make a list of LangChain documents with the content of those facts
documents = [Document(page_content=fact) for fact in creature_facts]
# Make a list of unique IDs for the vector database
uuids = [str(uuid4()) for _ in range(len(documents))]

# Store documents
vector_store.add_documents(documents=documents, ids=uuids)

# Initialize an empty list for session context
session = []
# Define a system prompt to guide behaviour
session.append(SystemMessage(content="""
You are an assistant for question-answering tasks.
Use the context provided by the user to answer the question.
If the context provided is irrelevant to the user question answer as if no context is provided.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
"""))


if __name__ == "__main__":
    print("Enter 'q' to exit")
    while True:
        # Initialize an empty string for retrieved info for each itteration
        retrieved_info = ""
        # User input
        query = input(">> ")

        if query == "q":
            break
        else:
            # Retrieve and show similar results
            results = vector_store.similarity_search_with_score(query, k=3)

            for res, score in results:
                if score < 0.4: # Filter retrieved info
                    print(f"== Retrieved Data Results ==\nScore: {score:3f}\nContent: {res.page_content}\n")
                    retrieved_info += res.page_content + " " # Store content

            if retrieved_info:
                human_message = HumanMessage(content=f"Answer the following question using the provided context DO NOT mention that you're referring to any context:\nContext: {retrieved_info}\nQuestion: {query}")
            else:
                human_message = HumanMessage(content=query)

            session.append(human_message)

            final_response = llm.invoke(session)
            session.append(AIMessage(content=final_response.content))
            print(final_response.content)

            # For streaming
            # res = llm.stream(session)
            # final_response = ""
            # for chunk in res:
            #     print(chunk.content, end="")
            #     final_response += chunk.content
            # session.append(AIMessage(content=final_response))