import os
import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os



index_name = "llama-rag-clean"
load_dotenv()

# Obtener las claves desde las variables de entorno
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
p_key = os.getenv("p_key")
pc = Pinecone(api_key=p_key)
from transformers import set_seed

set_seed(88)
from ctransformers import AutoConfig
config = AutoConfig.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML",context_length=4000 )

from ctransformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id="TheBloke/Llama-2-7B-Chat-GGML",hf=True,config=config,model_file="llama-2-7b-chat.ggmlv3.q2_K.bin",model_type="llama", gpu_layers=0)

model_id = "meta-llama/Llama-2-7b-chat-hf"

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id,use_auth_token=HF_AUTH_TOKEN,add_eos_token=True)

def llama_len(text):
    tokens = tokenizer(text, return_tensors="pt", return_attention_mask=False)["input_ids"][0]
    return len(tokens)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=llama_len,
    separators=["\n\n\n\n\n\n","\n\n\n\n","\n\n", "\n", " ", ""]
)

from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embed_model_id = "distiluse-base-multilingual-cased-v1"

device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={"device": device},
    encode_kwargs={"device": device, "batch_size": 32}
)

docs = [
    "documento numero uno",
    "otro documento más"
]

embeddings = embed_model.embed_documents(docs)

print(f"Tenemos {len(embeddings)} doc embeddings, cada uno con "
      f"una dimension de {len(embeddings[0])}.")

import time

index_name = "bm25"


index = pc.Index(index_name)
index.describe_index_stats()

import transformers, torch
from transformers import pipeline
from langchain import HuggingFacePipeline
generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,
    task="text-generation",
    do_sample=True,
    top_p=0.95,
    temperature=0.4,
    max_new_tokens=512,
    #eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=generate_text)

from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import (
    PineconeHybridSearchRetriever,
)

bm25 = BM25Encoder()

#################################################################
from langchain.load import dumps, loads
from typing import List, Any, Tuple
from pydantic import BaseModel
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFacePipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

class CustomRetriever(BaseRetriever):
    llm: HuggingFacePipeline
    vectorstore: Pinecone
    k: int = 3

    def __init__(self, llm: HuggingFacePipeline, vectorstore: Pinecone, **kwargs: Any):
        super().__init__(llm=llm, vectorstore=vectorstore, **kwargs)
        self.llm = llm
        self.vectorstore = vectorstore

    def generate_queries(self, query: str) -> List[str]:
        query = query.lower() 
        prompt_template = """Eres un asistente de modelo de lenguaje en español de IA. 
        Su tarea es generar diferentes versiones de la pregunta dada por el ususario en español para recuperar documentos relevantes de una base de datos vectorial. 
        Al generar múltiples perspectivas sobre la pregunta del usuario, su objetivo es ayudar al usuario a superar algunas de las limitaciones de la búsqueda de similitudes basada en la distancia.
        Proporcione estas preguntas separadas por nuevas líneas y numerar.Pregunta original: {question}"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["question"]
        )
        
        prompt_str = prompt_template.format(question=query)
        response = self.llm(prompt_str)

        questions = response.split("\n")
        #filtered_lines = [line.strip() for line in questions if line.strip()]
        filtered_lines = [line.strip().lower() for line in questions if line.strip()]
        filtered_lines.insert(0, query) 
        return filtered_lines[:2]        
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        queries = self.generate_queries(query)
        results = []
               
        retrieve = PineconeHybridSearchRetriever(
            embeddings=embed_model, sparse_encoder=bm25, index=index, top_k=self.k)
        
        for q in queries:
            bm25.fit(q)
            contexts= retrieve.invoke(q)
            results.append(contexts)
            
        combined_results = [item for sublist in results for item in sublist]
        unique_documents_set = set((tuple(doc.metadata.items()), doc.page_content) for doc in combined_results)
        unique_documents_list = [Document(metadata=dict(metadata), page_content=page_content) for metadata, page_content in unique_documents_set]
    
        return unique_documents_list

from operator import itemgetter
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA

llm = HuggingFacePipeline(pipeline=generate_text)

text_field = "text"
vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)

template = """
[INST] <<SYS>>
Responde las preguntas en español.
<</SYS>> 
{context}
{question} [/INST]
"""
prompt = PromptTemplate(input_variables=["context","question"], template=template)

qa = RetrievalQA.from_chain_type(
    llm=llm,chain_type="stuff",
    retriever=CustomRetriever(llm,vectorstore),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

import re
def retrieval(answer):
    bm25.fit(answer)
    response= qa.invoke(answer)
    output= response['result']
    output = re.sub(r'\[INST\].*?\[/INST\]', '', output, flags=re.DOTALL).strip()
    return output


