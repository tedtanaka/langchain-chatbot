import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter API key for Langsmith: ")

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

print("Translate 'hi!' from English to Italian:")
result = llm.invoke(messages)
print(result.content)

print()
#print("Entire result:")
#print(result)


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

# Load and chunk contents of the blog
rag_info = {
    "agent": {
        "url":"https://lilianweng.github.io/posts/2023-06-23-agent/",
        "question": "What is Task Decomposition?",
    },
    "mothman": {
        "url": "https://www.wboy.com/only-on-wboy-com/paranormal-w-va/the-legend-of-mothman-paranormal-w-va/",
        "question": "Who first saw the Mothman?",
    },
    "asus": {
        "url": "https://www.asus.com/microsite/motherboard/asus-motherboards-win11-ready",
        "question" : "Can a Windows PC with an ASUS M51AC motherboard be upgraded to Windows 11?",
    },
    "carroll": {
        "url": "https://www.poetryfoundation.org/poems/42916/jabberwocky/",
        "question" : "Who did gyre and gimble in the wabe?",
    },
}
rag_topic = "carroll"

print("Load contents of an Internet blog")
loader = WebBaseLoader(
    web_paths=(rag_info[rag_topic]["url"],),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

print("Split up the blog contents and store in a vector database")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


print("Ask LLM a question about the blog contents")
print("\nQuestion: " + rag_info[rag_topic]["question"])
print("\nAnswer:")

response = graph.invoke({"question": rag_info[rag_topic]["question"]})
print(response["answer"])