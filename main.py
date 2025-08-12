import os 
import numpy as np
import faiss #it is a vector search library it stoares and seacrh embeddings
from pypdf import PdfReader #reads pdf file
import google.generativeai as genai #it loads google gemini library

#1: Setup API Key
os.environ["GEMINI_API_KEY"] = "AIzaSyCNgvK8BWyY9ZxlMIfJ5I1q_6vzeoaZ078" #stores api key
genai.configure(api_key=os.environ["GEMINI_API_KEY"]) #configures gemini client

#2: Read PDF
def read_pdf(file_path):
    reader = PdfReader(file_path) #open pdf
    text = "" #empty string to accumulate pages
    for page in reader.pages: #loops through every page in the PDF.
        text += page.extract_text() + "\n" #extracts text from each page and appends it with a newline
    return text

#3: Chunk Text
def chunk_text(text, chunk_size=500): #function to split long text into smaller pieces
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

#4: Create Embeddings
def get_embedding(text): #function to call Gemini to get an embedding vector for a piece of text.
    embed = genai.embed_content( #API call to create an embedding. It returns an object
        model="models/embedding-001",
        content=text
    )
    return embed["embedding"] #extracts the numeric embedding

#5: Store in FAISS
def store_embeddings(chunks):
    embeddings = np.array([get_embedding(chunk) for chunk in chunks]).astype("float32") #creates a 2D NumPy array from all chunk embeddings
    dimension = embeddings.shape[1] #gets the length of each embedding vector
    index = faiss.IndexFlatL2(dimension) #creates a FAISS index that does exact nearest-neighbor search using L2 (Euclidean) distance
    index.add(embeddings) #inserts all embeddings into the index
    return index, chunks

#6: Search Function
def search(query, index, chunks, k=3): #function to find top-k chunks relevant to a query
    query_vector = np.array([get_embedding(query)]).astype("float32") #get the query embedding and format as a 2D NumPy array (shape [1, dim]) for FAISS
    _, I = index.search(query_vector, k) #FAISS returns distances and indices; _ receives distances (ignored here), I receives indices of nearest neighbors
    return [chunks[i] for i in I[0]] 

#7: Q&A Function
def answer_question(query, index, chunks):
    context = "\n".join(search(query, index, chunks)) #joins the retrieved chunks into one context string to give to the LLM
    prompt = f"Use this context to answer:\n{context}\n\nQuestion: {query}\nAnswer:" #builds a prompt instructing the model to use that context to answer the question
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt) #calls the Gemini generation endpoint with the prompt
    return response.text

#RUN THE APP
if __name__ == "__main__":
    # 📂 Read all PDFs from the data folder
    data_folder = "data"
    all_text = ""

    for file in os.listdir(data_folder):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(data_folder, file)
            print(f"Reading: {file_path}")
            all_text += read_pdf(file_path) + "\n"

    # Chunk and store embeddings for ALL PDFs together
    chunks = chunk_text(all_text)
    index, chunks = store_embeddings(chunks)


    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = answer_question(query, index, chunks)
        print("\nAnswer:", answer)
