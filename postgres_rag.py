from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

load_dotenv()

conn_str = os.getenv('NEON_CONN_STR')

def get_relevant_answers(prompt):
    with psycopg2.connect(conn_str) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query_embedding = ollama_embeddings_1024.embed_query(prompt)
            cur.execute("""
                        SELECT a, q_embeddings <=> %s::vector AS distance
                        FROM qa
                        WHERE q_embeddings <=> %s::vector < 0.4
                        ORDER BY distance
                        LIMIT 3;
                        """,
                        (query_embedding, query_embedding))
            rows = cur.fetchall()
            relevant_answers = [x['a'] for x in rows]
            print(f"Relevant Answers: {relevant_answers}", end='\n---\n')
    return relevant_answers

# Initialize Ollama models
ollama_embeddings_1024 = OllamaEmbeddings(model="embedding-cpu")
llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0,
)

# Initialize an empty list for session context
session = []
# Define a system prompt to guide behaviour
session.append(SystemMessage(content="""
You are an assistant for question-answering tasks that has access to external resources which are added along with the user question.
Use the resources provided by the user to answer the question.
If the context provided is irrelevant to the user question, answer as if no context is provided.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
"""))

if __name__ == "__main__":
    print("Enter 'q' to exit")
    while True:
        # User input
        query = input(">> ")

        if query == "q":
            break
        else:
            retrieved_info = get_relevant_answers(query)

            if retrieved_info:
                human_message = HumanMessage(content=f"Answer the following question using the provided context DO NOT mention that you're referring to any context:\nContext: {retrieved_info}\nQuestion: {query}")
            else:
                human_message = HumanMessage(content=query)

            session.append(human_message)

            final_response = llm.invoke(session)
            session.append(AIMessage(content=final_response.content))
            print(final_response.content)