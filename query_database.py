import argparse
# from dataclasses import dataclass
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.llms import HuggingFacePipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

CHROMA_PATH = "chroma"

# Prompt structure that is passed to the Model to get response
PROMPT_TEMPLATE = """Use only the context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=2)
    if len(results) == 0 or results[0][1] < 0.4:
        print(f"Unable to find matching results.")
        return

    # Generates the prompt using the template defined above
    MAX_CONTEXT_CHARS = 1200  # start here for flan-t5-base
    context_text = "\n\n---\n\n".join(doc.page_content for doc, _ in results)
    context_text = context_text[:MAX_CONTEXT_CHARS]
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = OllamaLLM(model="gemma3:1b") # load_local_llm() # chose to keep everything local instead of using OpenAI
    response_text = model.invoke(prompt) # removed predict since HuggingFace is not a ChatModel

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

# Local HuggingFace Model
# def load_local_llm():
#     model_id = "google/Mistral-7B-Instructc" # fast and instruction tuned

#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

#     pipe = pipeline(
#         "text2text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=128,
#         do_sample=False,   # deterministic
#     )

#     return HuggingFacePipeline(pipeline=pipe)

if __name__ == "__main__":
    main()