from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# import os

# os.environ['HF_HOME'] = 'D:/huggingface_cache'  # Set your desired cache directory
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.2,
        max_new_tokens=50
    )
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("What is the capital of France?")

print(response)