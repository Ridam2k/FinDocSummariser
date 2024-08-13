from typing import List
import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
)
#from langchain_community.llms import Ollama
from langchain.docstore.document import Document
from langchain_community.llms import Ollama

from langchain_community.chat_models import ChatOllama
from langchain.chat_models import ChatOpenAI

from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

import chainlit as cl
import streamlit as st

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100)


def extract_response(res):
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]
    text_elements = []  # type: List[cl.Text]
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
            source_names = [text_el.name for text_el in text_elements]
            if source_names:
                answer += f"\nSources: {', '.join(source_names)}"
            else:
                answer += "\nNo sources found"
    return cl.Message(content=answer, elements=text_elements)


@cl.on_chat_start
async def on_chat_start():
    files = None
    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    print(file)
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Read the PDF file
    pdf = PyPDF2.PdfReader(file.path)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()

    # Split the text into chunks
    texts = text_splitter.split_text(pdf_text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    # embeddings = OllamaEmbeddings(model="mistral")
    embeddings = OpenAIEmbeddings(
        openai_api_key="")         #insert key here
    docsearch = await cl.make_async(Chroma.from_texts)(texts, embeddings, metadatas=metadatas)

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history", output_key="answer", chat_memory=message_history, return_messages=True
    )

    general_system_template = r"""
    You are a financial document analysis expert, specialising in 10k reports. Given a specific context, comprehensively analyse the uploaded document to answer and include all tables and graphs wherever applicable to generate the response.
    ---- {context} ----
    """

    general_user_template = "Question:```{question}```"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]

    qa_prompt = ChatPromptTemplate.from_messages(messages)

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )

    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)

    # Sending an action button within a chatbot message
    actions = [
        cl.Action(name="Analyze Part I", value="part1",
                  description="Analyze Part I of the report"),

        cl.Action(name="Analyze Part II", value="part2",
                  description="Analyze Part II of the report"),
    ]
    await cl.Message(content="Select what content to analyze from the report:", actions=actions).send()


@cl.action_callback("Analyze Part I")
async def send_prompt(action: cl.Action):
    # Sending two more action buttons within a chatbot message
    actions = [
        cl.Action(name="Summarise Part I", value="summarisePart1",
                  description="Summarise Part I of the report"),
        cl.Action(name="Business Overview", value="businessOverview",
                  description="Get an overview of the business"),
        cl.Action(name="Risk Factors", value="riskFactors",
                  description="Get an overview of the rsik factors"),
    ]
    await cl.Message(content="Select which sections to query:", actions=actions).send()


@cl.action_callback("Analyze Part II")
async def send_prompt(action: cl.Action):
    # Sending two more action buttons within a chatbot message
    actions = [
        cl.Action(name="Summarise Part II", value="summarisePart2",
                  description="Summarise Part II of the report"),
        cl.Action(name="Consolidated Balance Sheet Statement", value="consBalSheet",
                  description="Get insights from the Consolidated Balance Sheet Statement"),
    ]
    await cl.Message(content="Select which sections to query:", actions=actions).send()


@cl.action_callback("Summarise Part I")
async def send_prompt_1(action: cl.Action):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    prompt = "What are the main topics covered in Part I of the report? Mention the key takeaways. "
    message_doc = cl.Message(content=prompt)

    res = await chain.ainvoke(message_doc.content, callbacks=[cb])
    output = extract_response(res)

    await output.send()
    return "Hardcoded prompt 1 processed."


@cl.action_callback("Business Overview")
async def send_prompt_2(action: cl.Action):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    prompt = "What are the main topics covered in the business section in Part I of the report? Mention the key takeaways"
    message_doc = cl.Message(content=prompt)

    res = await chain.ainvoke(message_doc.content, callbacks=[cb])
    output = extract_response(res)

    await output.send()
    return "Hardcoded prompt 2 processed."


@cl.action_callback("Summarise Part II")
async def send_prompt_1(action: cl.Action):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    # Replace this with your hardcoded prompt 1
    prompt = "What are the main topics covered in Part II of the report? Mention the key takeaways. "
    message_doc = cl.Message(content=prompt)

    res = await chain.ainvoke(message_doc.content, callbacks=[cb])
    output = extract_response(res)

    await output.send()
    return "Hardcoded prompt 1 processed."


@cl.action_callback("Consolidated Balance Sheet Statement")
async def send_prompt_2(action: cl.Action):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    prompt = "Discuss the company's consolidated balance sheet. What are the key numbers in this table and what do they represent?"
    message_doc = cl.Message(content=prompt)

    res = await chain.ainvoke(message_doc.content, callbacks=[cb])
    output = extract_response(res)

    await output.send()
    return "Hardcoded prompt 2 processed."


@cl.action_callback("Risk Factors")
async def send_prompt_2(action: cl.Action):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    prompt = "Refer to Item 1A Risk Factors of the report and summarize all the risks specific to the company mentioned in it. Include all the risks mentioned."
    message_doc = cl.Message(content=prompt)

    res = await chain.ainvoke(message_doc.content, callbacks=[cb])
    output = extract_response(res)

    await output.send()
    return "Hardcoded prompt 2 processed."


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()
    message_doc = cl.Message(content=message.content)

    res = await chain.ainvoke(message_doc.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]
    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
