from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.fake import FakeListLLM

def create_chain():
    # Create a mock LLM for testing
    responses = ["This is a mock response for: {}".format]
    llm = FakeListLLM(responses=responses)
    
    # Create prompt template
    template = """Question: {question}
    
    Answer: Let me help you with that."""
    
    prompt = PromptTemplate(
        input_variables=["question"],
        template=template
    )
    
    # Create and return the chain
    return LLMChain(llm=llm, prompt=prompt)

def main():
    # Initialize the chain
    chain = create_chain()
    
    # Test question
    question = "What is LangChain?"
    
    # Run the chain
    response = chain.run(question=question)
    print(f"Q: {question}")
    print(f"A: {response}")

if __name__ == "__main__":
    main()