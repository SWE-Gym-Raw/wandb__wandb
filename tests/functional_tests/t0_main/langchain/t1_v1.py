import datetime

from langchain.chains import LLMChain
from langchain.llms.fake import FakeListLLM
from langchain.prompts import PromptTemplate
from wandb.integration.langchain import WandbTracer


def simple_fake_test():
    llm = FakeListLLM(responses=[f"Fake response: {i}" for i in range(100)])

    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    for i in range(10):
        chain(f"q: {i} - {datetime.datetime.now().timestamp()}")


WandbTracer.init()
simple_fake_test()
WandbTracer.finish()
