from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv
from datetime import datetime

# Carrega variáveis de ambiente
load_dotenv()

# Função para extrair texto dos PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Função para obter a cadeia de conversa
def get_conversational_chain(api_key=None):
    print("\n")
    system = """
    Você é um assistente RAG. Responda da maneira mais detalhada possível pelo contexto entregue,
    se a pergunta não estiver no contexto apenas diga "Pergunta não disponível no contexto", não diga respostas erradas. \n
    """
    
    system = """
    Você é um assistente RAG. Voce pode responder de maneira ampla ao perguntado. \n
    """
    
    prompt_template = f"""
    {system}
    Contexto:\n {{context}}?\n
    Pergunta: \n{{question}}\n
    Resposta:
    """
    print(prompt_template)
    print("\n")
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["question", "context"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

# Função principal de interação
def user_input(user_question, api_key, pdf_docs, conversation_history):
    
    text_chunks = get_text_chunks(get_pdf_text(pdf_docs))
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(vectorstore=new_db, api_key=api_key)
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    response_output = response['output_text']
    
    # Exibindo resultados
    print(f"Pergunta: {user_question}")
    print(f"Resposta: {response_output}")

    # Adicionando à história da conversa
    conversation_history.append((user_question, response_output, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

# Entrada principal no terminal
def main():
    conversation_history = []
    
    # Colocar o caminho do arquivo PDF aqui
    pdf_docs = ["samsung-s23.pdf"]  # Exemplo de lista de PDFs
    
    # Chave da API do Google
    api_key = os.getenv('Gemini')
    
    while True:
        user_question = input("Faça uma pergunta: ")
        if user_question.lower() == "sair":
            break
        user_input(user_question, api_key, pdf_docs, conversation_history)

if __name__ == "__main__":
    main()
