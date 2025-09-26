import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI


# Disable Chroma telemetry
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
temp=float(os.getenv("OPENAI_TEMPERATURE", 0))
api_key=os.getenv("OPENAI_API_KEY")
anth_api = os.getenv("ANTHROPIC_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

global llm
llm = ChatOpenAI(
    model_name=model,
    temperature=temp,
    openai_api_key=api_key
)

def chunking(text: str):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain_community.vectorstores import Chroma
    from langchain.schema import Document   

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    langchain_embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Reduced overlap to minimize duplicate tokens
    splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

    documents = [Document(page_content=text)]
    docs = splitter.split_documents(documents)

    global vectordb, qa_chain
    vectordb = Chroma.from_documents(docs, langchain_embeddings)

    # Reduced retriever size
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",  # avoids dumping all chunks at once
        retriever=retriever,
        return_source_documents=False
    )
    return qa_chain

def proposal_change(query, block_html,action):
    if action == "add":
        instruction = f"""
        You are an expert HTML editor.
        Here is the HTML block:
        {block_html}

        User request: {query}

        Task: ADD the new content inside this block (e.g., append new list items if it's a list).
        Keep existing structure intact.
        Return ONLY the updated block.
        """
    elif action == "remove":
        instruction = f"""
        You are an expert HTML editor.
        Here is the HTML block:
        {block_html}

        User request: {query}

        Task: REMOVE the specified content from this block.
        Keep the rest unchanged.
        Return ONLY the updated block.
        """
    else:  # update
        instruction = f"""
        You are an expert HTML editor.
        Here is the HTML block:
        {block_html}

        User request: {query}

        Task: UPDATE this block according to the request.
        Keep structure intact.
        Return ONLY the updated block.
        """
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        messages=[{"role": "user", "content": instruction}],
    )
    return resp.choices[0].message.content.strip()